import os
from typing import List, Tuple
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_jina import JinaEmbeddings
from langchain_chroma import Chroma
from langchain_postgres.vectorstores import PGVector
# from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

class RetrieverPipeline:
    def __init__(self, db_type="chroma", embedding_type="openai"):
        self.db_type = db_type
        self.embedding_type = embedding_type
        self._initialize_embeddings()

        # 初始化数据库连接
        if db_type == "chroma":
            collection_name = f"{embedding_type}_chroma_collection"
            db_path = f"./corpus/vector_store/{embedding_type}/chroma_db"
            self.vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=db_path,
            )
        elif db_type == "pgvector":
            connection =  "postgresql+psycopg://langchain:langchain@localhost:6024/langchain" # TODO: change to your own db
            collection_name = f"{embedding_type}_pgvector_collection"
            db_path = f"./corpus/vector_store/{embedding_type}/pgvector_db"
            self.vector_store = PGVector(
                embeddings=self.embeddings,
                collection_name=collection_name,
                connection=connection,
                use_jsonb=True,
            )
        else:
            raise ValueError("Unsupported database type")

    def _initialize_embeddings(self):
        if self.embedding_type == "openai":
            # text-embedding-3-large
            self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))
        
        elif self.embedding_type == "azure_openai":
            api_key = os.getenv("AZURE_API_KEY")
            azure_endpoint = os.getenv("AZURE_ENDPOINT")
            azure_api_version = os.getenv("OPENAI_API_VERSION")
            self.embeddings = AzureOpenAIEmbeddings(model="text-embedding-3-large", api_version=azure_api_version, azure_endpoint=azure_endpoint, api_key=api_key)

        elif self.embedding_type == "jina":
            self.embeddings =JinaEmbeddings(model_name="jina-embeddings-v2-base-en", jina_api_key=os.getenv("GINA_API_KEY"))

        elif self.embedding_type == "sentence_transformer":
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        else:
            raise ValueError("Unsupported embedding type")
        

    def retrieve_documents(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        执行检索并返回结果。

        Args:
            query (str): 查询文本
            k (int): 返回最相关的文档数量

        Returns:
            List[Tuple[str, float]]: 文档内容和对应的分数
        """
        print(f"正在使用 {self.db_type} 数据库和 {self.embedding_type} 模型进行检索...")
        if self.db_type == "chroma":
            results = self.vector_store.similarity_search_with_score(query, k=k)
        elif self.db_type == "pgvector":
            query_embedding = self.embeddings.embed_query(query)
            results = self.vector_store.similarity_search(query_embedding, k=k)
        else:
            raise ValueError("Unsupported database type")

        for doc in results:
            print(f"* {doc.page_content} [{doc.metadata}]")

        formatted_results = [(doc.page_content, score) for doc, score in results]
        return formatted_results

# Example usage
if __name__ == "__main__":
    # 配置参数
    db_type = "chroma"  # 选择 'chroma' 或 'pgvector'
    embedding_type = "openai"  # 选择 'openai', 'azure_openai', 或 'sentence_transformer'
    collection_name = "retrieval_collection"
    db_path = "./corpus/vector_store/chroma_db"  # Chroma 的路径
    pg_config = {
        "dbname": "mydatabase",
        "user": "myuser",
        "password": "mypassword",
        "host": "localhost",
    }

    # 初始化检索器
    retriever = RetrieverPipeline(
        db_type=db_type,
        collection_name=collection_name,
        db_path=db_path,
        pg_config=pg_config if db_type == "pgvector" else None,
        embedding_type=embedding_type
    )

    # 执行检索
    query = "细胞膜的结构"  # 示例查询
    results = retriever.retrieve_documents(query, k=5)

    # 打印结果
    print("\n检索结果：")
    for idx, (content, score) in enumerate(results, start=1):
        print(f"结果 {idx} (分数: {score:.4f}):\n{content}\n")

