import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple
import asyncio

from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import JinaEmbeddings
from langchain_chroma import Chroma
from openai import OpenAI
from psycopg2 import connect
import time

@dataclass
class SearchResult:
    answer_text: str
    references: str

class RAGProcessor:
    def __init__(self, db_type="chroma", retriever_type="openai", pg_config=None):
        self.retriever_type = retriever_type
        self.db_type = db_type
        self.pg_config = pg_config
        
        if retriever_type == "openai":
            self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        elif retriever_type == "jina":
            self.embeddings = JinaAIEmbedding(model_name="TransformerTorchEncoder")
        else:
            raise ValueError("Unsupported retriever type")

        if db_type == "chroma":
            self.vector_store = Chroma(
                collection_name="cell_biology_collection_5th",
                embedding_function=self.embeddings,
                persist_directory="/root/moobius_ccs/resources/chroma_langchain_db",
            )
        elif db_type == "pgvector":
            self.conn = connect(**pg_config)
            self.cur = self.conn.cursor()
            self.cur.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    id SERIAL PRIMARY KEY,
                    retriever TEXT,
                    title TEXT,
                    embedding VECTOR(1536)
                )
                """
            )
            self.conn.commit()
        else:
            raise ValueError("Unsupported database type")

        self.client = OpenAI()

    async def process_query(self, query: str) -> SearchResult:
        textbook_search = await asyncio.gather(self._search_textbook(query))
        merged_metadata, page_contents_string = self._process_textbook_results(textbook_search)
        answer = self._generate_answer(query, page_contents_string)
        formatted_references = self._format_references(merged_metadata)
        return SearchResult(answer_text=answer, references=formatted_references)

    async def _search_textbook(self, query: str) -> List[Tuple]:
        if self.db_type == "chroma":
            return await asyncio.to_thread(
                self.vector_store.similarity_search_with_score, query, k=4, filter={"source": "textbook"}
            )
        elif self.db_type == "pgvector":
            query_embedding = self.embeddings.embed([query])[0]
            self.cur.execute(
                """
                SELECT title, embedding <-> %s AS distance
                FROM embeddings
                ORDER BY distance ASC
                LIMIT 4
                """,
                (query_embedding.tolist(),)
            )
            return [(row[0], row[1]) for row in self.cur.fetchall()]

    def _process_textbook_results(self, results: List[Tuple]) -> Tuple[dict, str]:
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
        prompt_template = f"""
        你是一位专业的课程答疑助手。请根据提供的课本内容用中文markdown格式为学生解答问题。

        问题: {query}
        课本参考内容: {page_contents_string}
        """
        completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt_template}]
        )
        return completion.choices[0].message.content

    def _format_references(self, references: Dict) -> str:
        formatted_str = ""
        for chapter, chapter_data in references.items():
            formatted_str += f"章节：{chapter}\n"
            for section_key, sections in chapter_data.items():
                if section_key == "sections":
                    for section, subsections in sections.items():
                        formatted_str += f"  └─ {section}\n"
                        for subsection in subsections:
                            if subsection:
                                formatted_str += f"      └─ {subsection}\n"
        return formatted_str

if __name__ == "__main__":
    async def main():
        pg_config = {
            "dbname": "vector_db",
            "user": "postgres",
            "password": "password",
            "host": "localhost",
            "port": "5432",
        }
        processor = RAGProcessor(db_type="pgvector", retriever_type="openai", pg_config=pg_config)
        test_query = "细胞膜的结构"
        print("问题:", test_query)

        start_time = time.time()
        result = await processor.process_query(test_query)
        end_time = time.time()

        print("回答:", result.answer_text)
        print("\n参考内容:")
        print(result.references)
        print(f"处理耗时: {end_time - start_time:.2f} 秒")

    asyncio.run(main())

