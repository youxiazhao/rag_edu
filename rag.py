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

from dotenv import load_dotenv
load_dotenv()


@dataclass
class SearchResult:
    answer_text: str
    references: str
    image_references: str

class RAGProcessor:
    def __init__(self, pg_config=None):
        self.embeddings = JinaEmbeddings(model_name="jina-embeddings-v3", jina_api_key=os.getenv("GINA_API_KEY"))
        connection = f"postgresql+psycopg://{pg_config['user']}:{pg_config['password']}@{pg_config['host']}:{pg_config['port']}/{pg_config['dbname']}"
        self.text_vector_store = PGVector(
                embeddings=self.embeddings,
                collection_name="rag_text_embeddings",
                connection=connection,
                use_jsonb=True
            )
        self.image_vector_store = PGVector(
                embeddings=self.embeddings,
                collection_name="rag_image_embeddings",
                connection=connection,
                use_jsonb=True
            )
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


    def process_query(self, query: str) -> SearchResult:
        # Perform similarity search for text
        text_results = self.text_vector_store.similarity_search_with_relevance_scores(query, k=5)
        print(text_results)
        # Perform similarity search for images
        image_results = self.image_vector_store.similarity_search_with_relevance_scores(query, k=3)
        print(image_results)
    
        # Process the text results
        merged_metadata, page_contents_string = self.process_textbook_results(text_results)
        
        # Process the image results
        image_references = self.process_image_results(image_results)
        
        # Generate the answer using both text and image content
        answer = self._generate_answer(query, page_contents_string, image_references)
        
        # Format the references
        formatted_references = self.format_references(merged_metadata)
        
        return SearchResult(answer_text=answer, references=formatted_references, image_references=image_references)


    def process_textbook_results(self, results: List[Tuple]) -> Tuple[dict, str]:
        merged_metadata = defaultdict(lambda: {"sections": defaultdict(set)})
        page_contents_string = ""
        
        for res, score in results:
            page_content = res.page_content
            book_title = res.metadata.get("book")
            chapter = res.metadata.get("chapter")
            section = res.metadata.get("section")
            subsection = res.metadata.get("subsection")
            
            page_contents_string += (
                f"书名: {book_title}\n"
                f"章节: {chapter}\n"
                f"节: {section}\n"
                f"小节: {subsection}\n"
                f"{page_content}\n\n"
            )
            if chapter and section:
                merged_metadata[chapter]["sections"][section].add(subsection)
                merged_metadata[chapter]["book_title"] = book_title  # Add book title to metadata

            # if chapter and section:
            #     merged_metadata[chapter]["sections"][section].add(subsection)
        
        return dict(merged_metadata), page_contents_string

    def process_image_results(self, results: List[Tuple]) -> str:
        base_url = "https://edurag.oss-cn-beijing.aliyuncs.com/images/"
        image_references = ""
        for res, score in results:
            book_title = res.metadata.get("book")
            image_path = res.metadata.get("image_path")
            full_image_url = f"{base_url}{image_path}"

            image_references += (
                f"书名: {book_title}\n\n",
                f"图片链接: {full_image_url}\n"
            )
        return image_references

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


if __name__ == "__main__":
    pg_config = {
        "dbname": "rag_db",
        "user": "edurag_user",
        "password": "edurag_pass",
        "host": "localhost",
        "port": 5432,
    } # TODO: change to your own pg_config

    processor = RAGProcessor(
        pg_config=pg_config,
    )

    test_query = "细胞膜的结构"
    print("问题:", test_query)

    start_time = time.time()
    result = processor.process_query(test_query)
    end_time = time.time()
    
    print("回答:", result.answer_text)
    print("\n参考内容:")
    print(result.references)
    print("\n参考图片:")
    print(result.image_references)

    print(f"处理耗时: {end_time - start_time:.2f} 秒")