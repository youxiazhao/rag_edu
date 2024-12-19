# rag_edu

# Setting Up PostgreSQL with PGVector Locally

This guide will walk you through installing PostgreSQL, configuring it with PGVector, and connecting to the database for vector-based operations.

## **1. Prerequisites**
- OS: MacOS/Linux/Windows (with WSL)
- PostgreSQL (version 16)
- `psycopg2` Python library (for database connections)

## **2. Installing PostgreSQL**

### MacOS (Homebrew):
```bash
brew install postgresql@16
brew services start postgresql@16
```

### Linux:
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

### Verify Installation:
```bash
psql --version
```

---

## **3. Setting Up the Database**

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

3. Exit PostgreSQL:
   ```sql
   \q
   ```

---

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

---

## **5. Configuring PostgreSQL Authentication**

1. Edit the `pg_hba.conf` file:
   ```bash
   sudo nano /etc/postgresql/16/main/pg_hba.conf
   ```

2. Update the following lines:
   ```plaintext
   # Allow local connections
   local   all             all                                     scram-sha-256
   host    all             all             127.0.0.1/32            scram-sha-256
   ```

3. Restart the PostgreSQL service:
   ```bash
   sudo systemctl restart postgresql
   ```

---

## **6. Connecting to PostgreSQL**

### Using Command Line:
To directly access the database, run the following command:
```bash
psql -U edurag_user -d edurag_db -h localhost -p 5432
```
You will be prompted to enter the password: `edurag_pass`.

### Example Python Script (using `psycopg2`):

```python
import psycopg2

# Connect to the database
conn = psycopg2.connect(
    dbname="edurag_db",
    user="edurag_user",
    password="edurag_pass",
    host="localhost",
    port="5432"
)

cur = conn.cursor()

# Create a table if not exists
cur.execute(f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        document_id VARCHAR(255) UNIQUE NOT NULL,
        book_title VARCHAR(255),
        embedding VECTOR(1024),
        embedding_type VARCHAR(50) NOT NULL,
        metadata JSONB
    )
""")
conn.commit()
```





