# 项目说明

本项目的目标是通过使用 Jina Embedding 和 PostgreSQL 的 pgvector 扩展，实现一个高效的查询和检索系统。以下是项目的整体逻辑：

1. 数据预处理
- 使用 Jina Embedding 对数据块进行嵌入处理。
- 根据不同的书籍以及图片/文字的维度，将嵌入结果存储到 PostgreSQL 的 pgvector 中。

2. 备选书单生成
- 数据库中所有的书籍都被列为一个备选列表。

3. Query处理
- 当收到一个Query时，首先使用 LLM1（大语言模型1）根据查询从备选书单中生成一个推荐书单。

4. Similarity计算
- 使用推荐书单中的书籍，从数据库中分别找到对应的文本和图片的embeddings。
- 计算查询嵌入与这些embeddings的相似度。
- 选出最相关的 top k 个文本文档和图片文档。

5. 答案生成
- 将选出的文本和图片文档作为输入提供给 LLM2（大语言模型2）。
- LLM2 根据参考资料回答查询，并提供参考来源和图片链接。


---

# Setting Up PostgreSQL with PGVector Locally

This guide will walk you through installing PostgreSQL, configuring it with PGVector, and connecting to the database for vector-based operations.

## 1. Prerequisites

- OS: MacOS/Linux/Windows (with WSL)
- PostgreSQL (version 16)
- `psycopg2` Python library (for database connections)

## 2. Installing PostgreSQL

### MacOS (Homebrew)

```bash
brew install postgresql@16
brew services start postgresql@16
```

### Linux

```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

### Verify Installation

```bash
psql --version
```

## 3. Setting Up the Database

1. Open PostgreSQL command line:

   ```bash
   psql -U postgres
   ```

2. Create a database and user:

   ```sql
   CREATE DATABASE edurag_db;
   CREATE USER edurag_user WITH PASSWORD 'edurag_pass';
   GRANT ALL PRIVILEGES ON DATABASE edurag_db TO edurag_user;
   ```

## **4. Installing PGVector Extension**

1. Clone the PGVector repository:
   ```bash
   git clone https://github.com/pgvector/pgvector.git
   cd pgvector
   ```

2. Compile and install PGVector:
   ```bash
   make
   sudo make install
   ```

3. Enable the extension in the database:
   ```bash
   psql -U postgres -d edurag_db
   CREATE EXTENSION IF NOT EXISTS vector;
   \q
   ```


## **6. Connecting to PostgreSQL**

### Using Command Line:
To directly access the database, run the following command:
```bash
psql -U edurag_user -d edurag_db -h localhost -p 5432
```
You will be prompted to enter the password: `edurag_pass`.



# 环境配置
1. requirements.txt 中列出了大多数需要的依赖包，使用 pip install -r requirements.txt 安装即可

2. 在.env文件中配置
OPENAI_API_KEY
JINA_API_KEY


# 各模块功能

1. data_chunk.py 中列出了数据处理的方法，可以调整chunk size和overlap size

2. embeddings.py 中列出了embeddings存储到pgvector的方法，可以调整embeddings的模型，数据库的存储设置，存储时的batch size等

```
self.cur.execute(f"""
   CREATE TABLE IF NOT EXISTS {table_name} (
      document_id VARCHAR(255) UNIQUE NOT NULL,
      book_title VARCHAR(255),
      embedding VECTOR(1024),
      embedding_type VARCHAR(50) NOT NULL,
      metadata JSONB
      )
""")
```

3. rag.py 中列出了核心的process过程，可以调整llm和prompt, receive书的个数，retrieve document top k的个数，answer及reference的格式，
TODO：image url 需要调整

4. main.py 中列出了run的入口，可以设置pg_config, query, book_list, table_name（文本/图片）

