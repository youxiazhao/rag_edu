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


def load_book_lists(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data['core_book_list'], data['supplementary_book_list']


def main(pg_config, query, book_list, retrieve_text=True, retrieve_images=True):
    # Initialize the RAGProcessor
    processor = RAGProcessor()

    # Connect to the database
    conn = connect_to_db(pg_config)

    if conn:
        # Call the process_query function
        result = processor.process_query(query, conn, book_list, retrieve_text=retrieve_text, retrieve_images=retrieve_images)

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
    core_book_list, supplementary_book_list = load_book_lists('booklist.json')



    # retrive all
    book_list = core_book_list + supplementary_book_list
    main(pg_config, query, book_list, retrieve_text=True, retrieve_images=True)

   