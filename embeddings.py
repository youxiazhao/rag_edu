from langchain_community.embeddings import JinaEmbeddings
from transformers import AutoModel


from langchain_core.documents import Document
import psycopg2
import os
import json
from typing import List
from uuid import uuid4
from dotenv import load_dotenv
import time

load_dotenv()

class JSONLoader:
    def load_documents(self, file_path: str, doc_type: str = "text", test: bool = False, test_limit: int = 5) -> List[Document]:
        documents = []
        count = 0
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if test and count >= test_limit:
                    break
                item = json.loads(line.strip())
                
                if doc_type == "text":
                    metadata = {
                        "book": item.get("book"),
                        "chapter": item.get("chapter"),
                        "section": item.get("section"),
                        "subsection": item.get("subsection")
                    }
                    content = item["content"]
                elif doc_type == "image":
                    metadata = {
                        "book": item.get("book"),
                        "image_path": item.get("image_path")
                    }
                    content = item["description"]  # Assuming image data is stored in this key

                documents.append(Document(page_content=content, metadata=metadata))
                count += 1
        print(f"Loaded {len(documents)} documents from {file_path}")
        return documents

class EmbeddingPipeline:
    def __init__(self):
        try:
            self.embeddings = JinaEmbeddings(model_name="jina-embeddings-v3", jina_api_key=os.getenv("JINA_API_KEY"))
            self.conn = psycopg2.connect(
                dbname="rag_db",
                user="edurag_user",
                password="edurag_pass",
                host="localhost",
                port="5432"
            )
            self.cur = self.conn.cursor() 
        except Exception as e:
            print(f"Initialization error: {e}")
            self.cur = None 
        
        self.processed_files_path = "corpus/processed_files.json"
        self.processed_files = self.load_processed_files()

    def load_processed_files(self):
        if os.path.exists(self.processed_files_path):
            with open(self.processed_files_path, "r") as f:
                return set(json.load(f))
        return set()
    
    def save_processed_file(self, file_path):
        self.processed_files.add(file_path)
        with open(self.processed_files_path, "w") as f:
            json.dump(list(self.processed_files), f)

    def get_table_name(self, doc_type: str) -> str:
        # Define a mapping from document type to table name
        table_name_mapping = {
            "text": "rag_text_embeddings",
            "image": "rag_image_embeddings"
            # Add more mappings if there are other document types
        }
        return table_name_mapping.get(doc_type, "default_table_name")


    def create_table_if_not_exists(self, doc_type: str):
        if not self.cur:
            print("Database cursor is not initialized.")
            return
        table_name = self.get_table_name(doc_type)
        try:
            self.cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                document_id VARCHAR(255) UNIQUE NOT NULL,
                book_title VARCHAR(255),
                embedding VECTOR(1024),
                embedding_type VARCHAR(50) NOT NULL,
                metadata JSONB
            )
            """)
            self.conn.commit()
        except Exception as e:
            print(f"Error creating table {table_name}: {e}")

    def load_and_embed_documents(self, documents: List[Document], doc_type: str):
        if not self.cur:
            print("Database cursor is not initialized.")
            return
        table_name = self.get_table_name(doc_type)
        self.create_table_if_not_exists(doc_type)
        uuids = [str(uuid4()) for _ in documents]
        document_contents = [doc.page_content for doc in documents]
        embeddings = self.embeddings.embed_documents(document_contents)
        
        for document, embedding, uuid in zip(documents, embeddings, uuids):
            document_id = uuid
            book_title = document.metadata.get("book")
            metadata_json = json.dumps(document.metadata)  # Convert metadata to JSON string
  
            try:
                self.cur.execute(f"""
                    INSERT INTO {table_name} (document_id, book_title, embedding, embedding_type, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (document_id) DO NOTHING
                """, (document_id, book_title, embedding, doc_type, metadata_json))
            except Exception as e:
                print(f"Error inserting document {book_title} into table {table_name}: {e}")

        self.conn.commit()
        print("Transaction committed.")

    
    def embedding_pipeline(self, file_dir: str, doc_type: str = "text", test: bool = False, batch_size: int = 500):
        print(f"Embedding pipeline started for {file_dir} with {doc_type} documents")
        loader = JSONLoader()
        count = 0

        for file in os.listdir(file_dir):
            file_path = os.path.join(file_dir, file)
            
            for doc in os.listdir(file_path):
                count += 1
                if test and count >= 5:
                    break

                all_documents = []
                if doc.endswith(".jsonl"):
                    doc_path = os.path.join(file_path, doc)
                    if doc_path in self.processed_files:
                        print(f"Skipping already processed file: {doc_path}")
                        continue
                    try:
                        documents = loader.load_documents(doc_path, doc_type=doc_type, test=test)
                        all_documents.extend(documents)
                    except Exception as e:
                        print(f"Error loading documents from {doc_path}: {e}")
                        continue

                for i in range(0, len(all_documents), batch_size):
                    batch = all_documents[i:i + batch_size]
                    self.load_and_embed_documents(batch, doc_type)
                
                # Mark this file as processed
                self.save_processed_file(doc_path)

if __name__ == "__main__":
    # Initialize the embedding pipeline
    pipeline = EmbeddingPipeline()
    
    # text
    file_directory = "./corpus/text_chunk"
    pipeline.embedding_pipeline(file_dir=file_directory, doc_type="text", test=False, batch_size=500)

    # image
    # file_directory = "./corpus/image_chunk"
    # pipeline.embedding_pipeline(file_dir=file_directory, doc_type="image", test=False, batch_size=500)
    
    # Close database connection
    pipeline.cur.close()
    pipeline.conn.close()