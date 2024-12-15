import os
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from uuid import uuid4
from datetime import datetime
import argparse
import tiktoken
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import JinaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from psycopg2 import connect
from langchain_azure_openai import AzureOpenAIEmbeddings
from sentence_transformers import SentenceTransformer


class EmbeddingManager:
    def __init__(self, vector_store, db_type, embedding_model):
        self.vector_store = vector_store
        self.db_type = db_type
        self.embedding_model = embedding_model

    def load_and_embed_documents(self, documents: List[Document]):
        uuids = [str(uuid4()) for _ in range(len(documents))]
        embeddings = self.embedding_model.embed([doc.page_content for doc in documents])
        self.vector_store.add_documents(documents=documents, ids=uuids, embeddings=embeddings)
        return len(documents)


class JSONLoader:
    def load_documents(self, file_dir: str) -> List[Document]:
        documents = []
        for file in os.listdir(file_dir):
            if file.endswith(".jsonl"):
                file_path = os.path.join(file_dir, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        item = json.loads(line.strip())
                        metadata = {
                            "source": item['id'],
                            "chapter": item["chapter"],
                            "section": item["section"],
                            "subsection": item["subsection"]
                        }
                        documents.append(Document(page_content=item["contents"], metadata=metadata))
        return documents


class EmbeddingPipeline:
    def __init__(self, db_type="chroma", embedding_type="openai"):
        self.db_type = db_type
        self.embedding_type = embedding_type
        self._initialize_embeddings()
        if db_type == "chroma":
            collection_name = f"{embedding_type}_chroma_collection"
            db_path = f"./corpus/vector_store/{embedding_type}/chroma_db"
            self.vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=db_path,
            )
        elif db_type == "pgvector":
            connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
            collection_name = f"{embedding_type}_pgvector_collection"
            db_path = f"./corpus/vector_store/{embedding_type}/pgvector_db"
            self.vector_store = PGVector(
                embeddings=self.embeddings,
                collection_name=collection_name,
                connection=connection,
                use_jsonb=True,
            )
            

    def _initialize_embeddings(self):
        if self.embedding_type == "openai":
            self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=os.environ.get("OPENAI_API_KEY"))
        
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



    def embed_documents(self, file_dir: str):
        loader = JSONLoader()
        documents = loader.load_documents(file_dir)
        manager = EmbeddingManager(
            vector_store=self.vector_store,
            db_type=self.db_type,
            collection_name=self.collection_name,
            embedding_model=self.embeddings,
        )
        num_docs = manager.load_and_embed_documents(documents)
        print(f"{self.db_type.capitalize()}_{self.embedding_type.capitalize()}: Embedded {num_docs} documents from {file_dir}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Document Loader')
    parser.add_argument('--db_type', type=str, choices=['chroma', 'pgvector'], required=True, help='Database type')
    parser.add_argument('--embedding_type', type=str, choices=['openai', 'azure_openai', 'jina', 'sentence_transformer'], required=True, help='Embedding type')
    parser.add_argument('--db_path', type=str, default='./db', help='Database path for Chroma')
    parser.add_argument('--file_path', type=str, required=True, help='Path to the document file')
    args = parser.parse_args()


    pipeline = EmbeddingPipeline(
        db_type=args.db_type,
        db_path=args.db_path,
        embedding_type=args.embedding_type
    )
    pipeline.embed_documents(args.file_path)

if __name__ == "__main__":
    file_dir = "./data"  # Replace with your file directory
    embedding_type = "openai"  # Choose from ['openai', 'azure_openai', 'jina', 'sentence_transformer']
    db_type = "chroma"  # Choose from ['chroma', 'pgvector']
    collection_name = "my_collection"
    db_path = "./my_db"

    pipeline = EmbeddingPipeline(
        db_type=db_type,
        db_path=db_path,
        embedding_type=embedding_type
    )
pipeline.embed_documents(file_dir)
