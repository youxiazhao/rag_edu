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

class RAGProcessor:
    def __init__(self, db_type="chroma", embedding_type="openai", pg_config=None):
        self.db_type = db_type
        self.embedding_type = embedding_type
        self.embeddings = self._initialize_embeddings()
        self.vector_store = self._initialize_vector_store(pg_config)
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def _initialize_embeddings(self):
        if self.embedding_type == "openai":
            # text-embedding-3-large
            return OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))
        
        elif self.embedding_type == "azure_openai":
            api_key = os.getenv("AZURE_API_KEY")
            azure_endpoint = os.getenv("AZURE_ENDPOINT")
            azure_api_version = os.getenv("OPENAI_API_VERSION")
            return AzureOpenAIEmbeddings(model="text-embedding-3-large", api_version=azure_api_version, azure_endpoint=azure_endpoint, api_key=api_key)

        elif self.embedding_type == "jina":
            return JinaEmbeddings(model_name="jina-embeddings-v2-base-en", jina_api_key=os.getenv("GINA_API_KEY"))

        # elif self.embedding_type == "sentence_transformer":
        #     return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        else:
            raise ValueError("Unsupported embedding type")
        

    def _initialize_vector_store(self, pg_config):
        if self.db_type == "chroma":
            return Chroma(
                collection_name=f"{self.embedding_type}_chroma_collection",
                embedding_function=self.embeddings,
                persist_directory=f"./corpus/vector_store/{self.embedding_type}/chroma_db"
            )
        elif self.db_type == "pgvector":
            if not pg_config:
                raise ValueError("pg_config is required for pgvector")
            connection = f"postgresql+psycopg://{pg_config['user']}:{pg_config['password']}@{pg_config['host']}:{pg_config['port']}/{pg_config['dbname']}"
            return PGVector(
                embeddings=self.embeddings,
                collection_name=f"{self.embedding_type}_pgvector_collection",
                connection=connection,
                use_jsonb=True
            )
        else:
            raise ValueError("Unsupported database type")


    def process_query(self, query: str) -> SearchResult:
        results = self.vector_store.similarity_search_with_relevance_scores(query, k=3)
    
        # Process the results to generate the answer
        merged_metadata, page_contents_string = self.process_textbook_results(results)
        answer = self._generate_answer(query, page_contents_string)
        
        # Format the references
        formatted_references = self.format_references(merged_metadata)
        
        return SearchResult(answer_text=answer, references=formatted_references)


    def process_textbook_results(self, results: List[Tuple]) -> Tuple[dict, str]:
        merged_metadata = defaultdict(lambda: {"sections": defaultdict(set)})
        page_contents_string = ""
        
        for res, score in results:
            page_content = res.page_content
            chapter = res.metadata.get("chapter")
            section = res.metadata.get("section")
            subsection = res.metadata.get("subsection")
            
            page_contents_string += (
                f"章节: {chapter}\n"
                f"节: {section}\n"
                f"小节: {subsection}\n"
                f"{page_content}\n\n"
            )
            
            if chapter and section:
                merged_metadata[chapter]["sections"][section].add(subsection)
        
        return dict(merged_metadata), page_contents_string


    def _generate_answer(self, query: str, page_contents_string: str) -> str:
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
                ("human", "问题: {query}\n\n课本参考内容: {page_contents_string}"),
            ]
        )

        chain = prompt | self.llm
        output =chain.invoke(
            {
                "query": query,
                "page_contents_string": page_contents_string,
            }
        )

        return output.content

        # prompt_template = f"""
        # 你是一位专业的课程答疑助手。请根据提供的课本内容，用中文markdown格式为学生解答问题。

        # 回答要求：
        # 1. 回答长度控制在300-800字
        # 2. 语言要专业准确，符合大学本科的学术水平
        # 3. 可以引用课本内容，但要用自己的语言重新组织表达
        # 4. 忽略课本文字中出现的"图xx-xx"的引用，因为这些原书图片无法获取
        # 5. 回答要有逻辑性和连贯性，避免简单罗列知识点
        # 6. 如果问题涉及多个方面，要分点回答，确保全面性

        # 问题: {query}

        # 课本参考内容: {page_contents_string}
        # """

        # completion = self.client.chat.completions.create(
        #     model="gpt-4o-mini",
        #     messages=[
        #         {"role": "user", "content": prompt_template}
        #     ]
        # )
        
        # return completion.choices[0].message.content


    def format_references(self, references: dict) -> str:
        formatted_str = ""
        for chapter, chapter_data in references.items():
            formatted_str += f"章节：{chapter}\n"
            for section_key, sections in chapter_data.items():
                if section_key == "sections":
                    for section, subsections in sections.items():
                        formatted_str += f"  └─ {section}\n"
                        for subsection in subsections:
                            if subsection:  # 只有当subsection不为空时才添加
                                formatted_str += f"      └─ {subsection}\n"
        return formatted_str


if __name__ == "__main__":
    db_type = "chroma"  # Choose from ['chroma', 'pgvector']
    embedding_type = "openai"  # Choose from ['openai', 'azure_openai', 'jina',]
    pg_config = {
        "dbname": "mydatabase",
        "user": "myuser",
        "password": "mypassword",
        "host": "localhost",
        "port": 5432,
    } # TODO: change to your own pg_config

    processor = RAGProcessor(
        db_type=db_type,
        embedding_type=embedding_type,
        pg_config=pg_config if db_type == "pgvector" else None,
    )

    test_query = "细胞膜的结构"
    print("问题:", test_query)

    start_time = time.time()
    result = processor.process_query(test_query)
    end_time = time.time()
    
    print("回答:", result.answer_text)
    print("\n参考内容:")
    print(result.references)

    print(f"处理耗时: {end_time - start_time:.2f} 秒")