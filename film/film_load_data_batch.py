import logging

from langchain.schema.document import Document

from utils import ew_embedding_util
from utils import ew_mysql_util


# Global variable start
db = ew_embedding_util.get_chroma_db("chroma_film")
mysql_conn = ew_mysql_util.get_mysql_conn()
logger = logging.getLogger(__name__)
# Global variable ends

def main():    
    # Load into chroma one by one
    load_documents()


def load_documents():
    """Read sql database and one by one insert into chroma"""
    cursor = mysql_conn.cursor()
    cursor.execute("SELECT film_id, title, description FROM film WHERE film_id <= 1000")
    
    for film_id, title, description in cursor.fetchall():
        str = f"Id: {film_id}\nTitle: {title}\nDescription: {description}\n"
        insert_item(film_id, str)

    cursor.close()
    mysql_conn.close()


def insert_item(id, payload):
    """Insert a new item """
    
    try:
        logger.debug(f"Payload [{payload}]")
        doc = Document(page_content=payload, metadata={"mysql_id": id, "source":"film"})
        chunks = ew_embedding_util.split_documents([doc])
        ew_embedding_util.add_to_chroma(chunks, db)

        logger.debug(f"Item inserted at id [{id}]")

    except Exception as e:
        logger.error(f"Insert Error: {e}")

if __name__ == "__main__":
    main()


