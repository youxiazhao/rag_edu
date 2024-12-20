from rag import *
import psycopg2

def connect_to_db(pg_config):
    try:
        conn = psycopg2.connect(
            dbname=pg_config["dbname"],
            user=pg_config["user"],
            password=pg_config["password"],
            host=pg_config["host"],
            port=pg_config["port"]
        )
        return conn
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return None



def main(pg_config, query, book_list, table_name):
    # Initialize the RAGProcessor
    processor = RAGProcessor()

    # Connect to the database
    conn = connect_to_db(pg_config)

    if conn:
        # Call the process_query function
        result = processor.process_query(query, conn, table_name, book_list)

        # Print the results
        print("回答:", result.answer_text)
        print("\n参考内容:")
        print(result.references)
        print("\n参考图片:")
        print(result.image_references)

        # Close the database connection
        conn.close()



if __name__ == "__main__":
    pg_config = {
        "dbname": "rag_db",
        "user": "edurag_user",
        "password": "edurag_pass",
        "host": "localhost",
        "port": 5432,
    } # TODO: change to your own pg_config

    query = "描述细胞膜的结构及其在细胞功能中的作用。"
    book_list = [
        '29_基础生态学_4_', '6_细胞生物学_5_', '生物信息学_Bioinformatics_and_Functional_Genomics_中文版',
        '23_植物生理学_8_', 'Developmental_Biologyｨ9ｩ', '杨荣武生物化学原理_3_', '20_普通动物学_4_',
        '7_Molecular_Biology_of_The_Cellｨ7ｩ', 'Human_Physiology_An_Integrated_Approachｨ8ｩ',
    ]

    # image
    table_name = "rag_text_embeddings"
    table_name = "rag_image_embeddings"

    main(pg_config, query, book_list, table_name)
