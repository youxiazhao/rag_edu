import os
import json
# from abc import ABC, abstractmethod
# from typing import List, Dict, Any
from typing import List
from uuid import uuid4
# from datetime import datetime
# import argparse
# import tiktoken
# from tiktoken import get_encoding
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_community.embeddings import JinaEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
# from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv

load_dotenv()

class EmbeddingManager:
    def __init__(self, vector_store, embedding_model):
        self.vector_store = vector_store
        self.embedding_model = embedding_model

    def load_and_embed_documents(self, documents: List[Document]):
        uuids = [str(uuid4()) for _ in range(len(documents))]
        self.vector_store.add_documents(documents=documents, ids=uuids)
        return len(documents)


class JSONLoader:
    def load_documents(self, file_path: str, test: bool = False, test_limit: int = 5) -> List[Document]:
        documents = []
        count = 0
        if test:
            print("Running in test mode")
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if count >= test_limit:
                        break
                    item = json.loads(line.strip())
                    metadata = {
                        "source": item['id'],
                        "chapter": item["chapter"],
                        "section": item["section"],
                        "subsection": item["subsection"]
                    }
                    documents.append(Document(page_content=item["content"], metadata=metadata))
                    # print(documents)
                    count += 1
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line.strip())
                    metadata = {
                        "source": item['id'],
                        "chapter": item["chapter"],
                        "section": item["section"],
                        "subsection": item["subsection"]
                    }
                    documents.append(Document(page_content=item["content"], metadata=metadata))
                    count += 1
                    print(count)
                    print(f"Loaded {len(documents)} documents")
                    print(file_path)
                    print(item["id"])
                    print(item["content"])
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
            connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain" # TODO: change to your own db
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
            # text-embedding-3-large
            self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))
        
        elif self.embedding_type == "azure_openai":
            api_key = os.getenv("AZURE_API_KEY")
            azure_endpoint = os.getenv("AZURE_ENDPOINT")
            azure_api_version = os.getenv("OPENAI_API_VERSION")
            self.embeddings = AzureOpenAIEmbeddings(model="text-embedding-3-large", api_version=azure_api_version, azure_endpoint=azure_endpoint, api_key=api_key)

        elif self.embedding_type == "jina":
            self.embeddings =JinaEmbeddings(model_name="jina-embeddings-v2-base-en", jina_api_key=os.getenv("GINA_API_KEY"))

        # elif self.embedding_type == "sentence_transformer":
        #     self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        else:
            raise ValueError("Unsupported embedding type")



    def embed_documents(self, file_dir: str, test: bool = False, batch_size: int = 2000):
        loader = JSONLoader()
        files = sorted(os.listdir(file_dir))

        if test:
            # Limit to processing only the first file in test mode
            files = files[:1]

        for file in files:
            if file.endswith(".jsonl"):
                file_path = os.path.join(file_dir, file)
                documents = loader.load_documents(file_path, test=test)
                
                # Process documents in batches
                for i in range(0, len(documents), batch_size):
                    batch = documents[i:i + batch_size]
                    manager = EmbeddingManager(
                        vector_store=self.vector_store,
                        embedding_model=self.embeddings
                    )
                    num_docs = manager.load_and_embed_documents(batch)
                    print(f"{self.db_type.capitalize()}_{self.embedding_type.capitalize()}: Embedded {num_docs} documents from {file_path} (Batch {i // batch_size + 1})")



if __name__ == "__main__":
    file_dir = "./corpus/chunk"  # Replace with your file directory
    embedding_type = "openai"  # Choose from ['openai', 'azure_openai', 'jina',]
    db_type = "chroma"  # Choose from ['chroma', 'pgvector']
    batch_size = 2000  # Set your desired batch size

    pipeline = EmbeddingPipeline(
        db_type=db_type,
        embedding_type=embedding_type
    )
    pipeline.embed_documents(file_dir, test=True, batch_size=batch_size)
