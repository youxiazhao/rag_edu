import os
from typing import List, Tuple
from collections import defaultdict
from dataclasses import dataclass
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_community.embeddings import JinaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
# from sentence_transformers import SentenceTransformer
import time
import numpy as np
import json

import psycopg2
from dotenv import load_dotenv
load_dotenv()


@dataclass
class SearchResult:
    answer_text: str = ""
    references: str = ""
    image_references: str = ""

class RAGProcessor:
    def __init__(self, pg_config=None):
        self.embeddings = JinaEmbeddings(model_name="jina-embeddings-v3", jina_api_key=os.getenv("GINA_API_KEY"))
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def get_selected_embeddings(self, conn, table_name, placeholders):
        embeddings = []  # Initialize the list before the loop
        metadata = []    # Initialize the list before the loop
        with conn.cursor() as cur:
            cur.execute(f"SELECT book_title, embedding, metadata FROM {table_name}")  # Assuming 'id' is the document ID
            rows = cur.fetchall()
            for row in rows:
                book_title, embedding_str, metadata_json = row
                if book_title in placeholders:
                    embedding = np.array(json.loads(embedding_str))
                    embeddings.append(embedding)  # Convert to numpy array if needed
                    metadata.append(metadata_json)
        return embeddings, metadata

    def calculate_cosine_similarity(self, query_embedding, embeddings):
        print("Query Embedding Shape:", query_embedding.shape)
        print("Embeddings Shape:", embeddings.shape)
        query_norm = np.linalg.norm(query_embedding)
        query_embedding_normalized = query_embedding / query_norm if query_norm != 0 else query_embedding

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_normalized = np.divide(embeddings, norms, where=norms != 0)

        cosine_similarities = np.dot(embeddings_normalized, query_embedding_normalized)
        return cosine_similarities

    def search_similar_embeddings_pgvector(self, query: str, conn, table_name: str, k=5):
        query_embedding = self.embeddings.embed_query(query)
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT book_title, embedding <-> %s AS distance
                FROM {table_name}
                ORDER BY distance ASC
                LIMIT %s
            """,
            (query_embedding.tolist(), k)
            )
            results = cur.fetchall()
        return [(row[0], row[1]) for row in results]

    def process_textbook_results(self, results: List[Tuple]) -> Tuple[dict, str]:
        merged_metadata = defaultdict(lambda: {"sections": defaultdict(set)})
        page_contents_string = ""
    
        for res, score in results:
            page_content = res.get("page_content", "")
            book_title = res.get("metadata", {}).get("book", "")
            chapter = res.get("metadata", {}).get("chapter", "")
            section = res.get("metadata", {}).get("section", "")
            subsection = res.get("metadata", {}).get("subsection", "")
            
            page_contents_string += (
                f"书名: {book_title}\n"
                f"章节: {chapter}\n"
                f"节: {section}\n"
                f"小节: {subsection}\n"
                f"{page_content}\n\n"
            )
            if chapter and section:
                merged_metadata[chapter]["sections"][section].add(subsection)
                merged_metadata[chapter]["book_title"] = book_title

            return dict(merged_metadata), page_contents_string
        
    def format_references(self, references: dict) -> str:
        formatted_str = ""
        for chapter, chapter_data in references.items():
            book_title = chapter_data.get("book_title", "Unknown Title")  # Retrieve the book title
            formatted_str += f"书名：{book_title}\n"  # Add the book title at the beginning
            # formatted_str += f"章节：{chapter}\n"
            for section_key, sections in chapter_data.items():
                if section_key == "sections":
                    for section, subsections in sections.items():
                        formatted_str += f"  └─ {section}\n"
                        # for subsection in subsections:
                        #     if subsection:  # 只有当subsection不为空时才添加
                        #         formatted_str += f"      └─ {subsection}\n"
        return formatted_str

    def process_image_results(self, results: List[Tuple]) -> str:
        base_url = "https://edurag.oss-cn-beijing.aliyuncs.com/images/" # TODO: change to your own oss url
        image_references = ""
        for res, score in results:
            book_title = res.get("metadata", {}).get("book", "")
            image_path = res.get("metadata", {}).get("image_path", "")
            full_image_url = f"{base_url}{image_path}"

            image_references += (
                f"书名: {book_title}\n"
                f"图片链接: {full_image_url}\n\n"
            )
        return image_references
    

    def suggest_book_title(self, query: str, book_list: List[str]) -> str:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一位专业的检索助手。请根据用户的查询，分析列表中哪些书中的内容可以帮助回答问题，并返回选定的书单。\n\n"
                    "回答要求：\n"
                    "1. 只从给定的list中选择书的名字,不用考虑其格式是否是书名 \n"
                    "2. 不要改动任何书名\n"
                    "2. 不要超过5本书\n"
                    "3. 书单中不要出现重复的书名\n"
                ),
                ("human", "问题: {query}\n\n书单: {book_list}"),
            ]
        )
        chain = prompt | self.llm
        output = chain.invoke(
            {"query": query, "book_list": book_list}
        )
        return output.content

    def _generate_answer(self, query: str, page_contents_string: str, image_references: str) -> str:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一位专业的课程答疑助手。请根据提供的课本内容，用中文markdown格式为学生解答问题。\n\n"
                    "回答要求：\n"
                    "1. 回答长度控制在300-800字\n"
                    "2. 语言要专业准确，符合大学本科的学术水平\n"
                    "3. 可以引用课本内容，但要用自己的语言重新组织表达\n"
                    "4. 忽略课本文字中出现的\"图xx-xx\"的引用，因为这些原书图片无法获取\n"
                    "5. 回答要有逻辑性和连贯性，避免简单罗列知识点\n"
                    "6. 如果问题涉及多个方面，要分点回答，确保全面性\n"
                ),
                ("human", "问题: {query}\n\n课本参考内容: {page_contents_string}\n\n图片参考内容: {image_references}"),
            ]
        )

        chain = prompt | self.llm
        output =chain.invoke(
            {
                "query": query,
                "page_contents_string": page_contents_string,
                "image_references": image_references,
            }
        )

        return output.content
    


    def process_query(self, query: str, conn, book_list: List[str], k=5, retrieve_text=True, retrieve_images=True) -> SearchResult:
        # Step 1: Suggest book titles
        suggested_books = self.suggest_book_title(query, book_list)
        print("Suggested Books:", suggested_books)  # Debugging output

        if retrieve_text:
            # Step 2: Retrieve embeddings for the suggested books
            embeddings_text, metadata_text = self.get_selected_embeddings(conn, "rag_text_embeddings", suggested_books)
            print("Embeddings Retrieved:", len(embeddings_text))  # Debugging output
            
            # Step 3: Calculate similarity
            query_embedding = self.embeddings.embed_query(query)
            similarities = self.calculate_cosine_similarity(query_embedding, embeddings_text)

             # Step 4: Rank results
            top_k_indices = np.argsort(similarities)[-k:][::-1]
            top_k_results = [(metadata_text[i], similarities[i]) for i in top_k_indices]
            print("Top K Results:", top_k_results)  # Debugging output

            # Step 5: Process results
            merged_metadata, page_contents_string = self.process_textbook_results(top_k_results)

            text_references = self.format_references(merged_metadata)    
            print(text_references)


        if retrieve_images:
            embeddings_image, metadata_image = self.get_selected_embeddings(conn, "rag_image_embeddings", suggested_books)
            print("Embeddings Retrieved:", len(embeddings_image))  # Debugging output

            # Step 3: Calculate similarity
            query_embedding = self.embeddings.embed_query(query)
            similarities = self.calculate_cosine_similarity(query_embedding, embeddings_image)
            print("Similarities Calculated:", similarities)  # Debugging output

            # Step 4: Rank results
            top_k_indices = np.argsort(similarities)[-k:][::-1]
            top_k_results = [(metadata_image[i], similarities[i]) for i in top_k_indices]
            print("Top K Results:", top_k_results)  # Debugging output

            # Step 5: Process results
            image_references = self.process_image_results(top_k_results)
            

        answer_text = self._generate_answer(query, page_contents_string=page_contents_string if retrieve_text else "", image_references=image_references if retrieve_images else "")
        print("Generated Answer:", answer_text)  # Debugging output


        if not embeddings_text and not embeddings_image:
            return print("没有设置相关内容retrieval")
        
        return SearchResult(answer_text=answer_text, references=text_references, image_references=image_references)

