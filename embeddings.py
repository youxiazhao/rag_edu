import os
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from uuid import uuid4
from datetime import datetime
import argparse
import tiktoken
from tiktoken import get_encoding
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
    def load_documents(self, file_dir: str, test: bool = False) -> List[Document]:
        documents = []
        count = 0
        if test:
            print("Running in test mode")
            test_file= os.path.join(file_dir, sorted(os.listdir(file_dir))[0])
            with open(test_file, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line.strip())
                    metadata = {
                        "source": item['id'],
                        "chapter": item["chapter"],
                        "section": item["section"],
                        "subsection": item["subsection"]
                    }
                    documents.append(Document(page_content=item["content"], metadata=metadata))
                    print(documents)
        else:
            for file in sorted(os.listdir(file_dir)):
                if file.endswith(".jsonl"):
                    file_path = os.path.join(file_dir, file)
                    if file_path == "./corpus/jina_chunk/20_普通动物学_4_200.jsonl":
                        continue
                    if file_path == "./corpus/jina_chunk/38_Lewin基因_12_432.jsonl":
                        continue
                    if file_path == "./corpus/jina_chunk/进化生物学基础_4_8.jsonl":
                        continue
                    if file_path == "./corpus/jina_chunk/33_动物行为学_2_115.jsonl":
                        continue
                    if file_path == "./corpus/jina_chunk/33_动物行为学_2_115_1.jsonl":
                        continue
                    if file_path == "./corpus/jina_chunk/行为生态学_2_195.jsonl":
                        continue
                    if file_path == "./corpus/jina_chunk/分子细胞生物学_3_102.jsonl":
                        continue
                    if file_path == "./corpus/jina_chunk/37_遗传学_从基因到基因组_6_316.jsonl":
                        continue
                    if file_path == "./corpus/jina_chunk/分子细胞生物学_3_263.jsonl":
                        continue
                    if file_path == "./corpus/jina_chunk/15_38.jsonl":
                        continue
                    if file_path == "./corpus/jina_chunk/38_Lewin基因_12_377.jsonl":
                        continue
                    if file_path == "./corpus/jina_chunk/23_植物生理学_8_100.jsonl":
                        continue

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

        elif self.embedding_type == "sentence_transformer":
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        else:
            raise ValueError("Unsupported embedding type")



    def embed_documents(self, file_dir: str, test: bool = False):
        loader = JSONLoader()
        documents = loader.load_documents(file_dir, test=test)
        manager = EmbeddingManager(
            vector_store=self.vector_store,
            embedding_model=self.embeddings
        )
        num_docs = manager.load_and_embed_documents(documents)
        print(f"{self.db_type.capitalize()}_{self.embedding_type.capitalize()}: Embedded {num_docs} documents from {file_dir}")



# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Document Loader')
#     parser.add_argument('--db_type', type=str, choices=['chroma', 'pgvector'], required=True, help='Database type')
#     parser.add_argument('--embedding_type', type=str, choices=['openai', 'azure_openai', 'jina', 'sentence_transformer'], required=True, help='Embedding type')
#     parser.add_argument('--db_path', type=str, default='./db', help='Database path for Chroma')
#     parser.add_argument('--file_path', type=str, required=True, help='Path to the document file')
#     args = parser.parse_args()


#     pipeline = EmbeddingPipeline(
#         db_type=args.db_type,
#         db_path=args.db_path,
#         embedding_type=args.embedding_type
#     )
#     pipeline.embed_documents(args.file_path)

if __name__ == "__main__":
    file_dir = "./corpus/jina_chunk"  # Replace with your file directory
    embedding_type = "jina"  # Choose from ['openai', 'azure_openai', 'jina', 'sentence_transformer']
    db_type = "chroma"  # Choose from ['chroma', 'pgvector']

    pipeline = EmbeddingPipeline(
        db_type=db_type,
        embedding_type=embedding_type
    )
    pipeline.embed_documents(file_dir, test=False)
