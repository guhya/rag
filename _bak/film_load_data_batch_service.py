import logging
import argparse
import mysql.connector

from langchain.schema.document import Document

from langchain_community.llms.ollama import Ollama
from langchain.prompts import ChatPromptTemplate
from utils import ew_embedding_util


# Global variable start
db = ew_embedding_util.get_chroma_db("chroma_film")
logger = logging.getLogger(__name__)
# Global variable ends

# Setup MySQL and Chroma DB
mysql_conn = mysql.connector.connect(
    host="localhost",
    user="sakila",
    password="sakila",
    database="sakila"
)

def main():

    # Check if the database should be cleared (using the --reset flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        logger.debug("clearing Database")
        ew_embedding_util.clear_chroma_db("chroma_film")

    # Create (or update) the data store.
    documents = load_documents()
    # print(f"Documents : {documents}")
    
    chunks = ew_embedding_util.split_documents(documents)
    # print(f"Chunks : {chunks}")
    
    ew_embedding_util.add_to_chroma(chunks)


def load_documents():

    cursor = mysql_conn.cursor()
    cursor.execute("SELECT film_id, title, description FROM film WHERE film_id < 5")
    
    docs = []
    for film_id, title, description in cursor.fetchall():
        str = f"Id: {film_id}\nTitle: {title}\nDescription: {description}\n"
        insert_item(film_id, str)

    cursor.close()
    mysql_conn.close()

    return docs

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

def generate_descriptions():

    PROMPT_TEMPLATE = """
    You will generate at least 3 paragraph of a movie description and synopsis.
    You do not need to style the headings.

    {context}

    Generate the response for the movie title: {question}
    """
    model = Ollama(model="llama3.1:8b-instruct-q8_0")

    cursor = mysql_conn.cursor()
    cursor.execute("SELECT film_id, title FROM film")
    
    update_sql = "UPDATE film SET description = %s WHERE film_id = %s"
    for film_id, title in cursor.fetchall():
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context="", question=title)

        response_text = model.invoke(prompt)
        str = f"Title: {title}\nDescription: {response_text}\n"
        logger.debug(str)

        try:
            cursor.execute(update_sql, (response_text, film_id))
            mysql_conn.commit()        
        except mysql.connector.Error as err:
            logger.error(f"Error: {err}")
            mysql_conn.rollback()

    cursor.close()
    mysql_conn.close()
