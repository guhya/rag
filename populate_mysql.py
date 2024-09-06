import argparse
import os
import shutil
import mysql.connector

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma

from langchain_community.llms.ollama import Ollama
from langchain.prompts import ChatPromptTemplate
from get_embedding_function import get_embedding_function


CHROMA_PATH = "chroma_mysql"
DATA_PATH = "data"

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
        print("✨ Clearing Database")
        clear_database()

    # generate_descriptions()
    # if 1==1: 
    #    return

    # Create (or update) the data store.
    documents = load_documents()
    # print(f"Documents : {documents}")
    
    print("\n\n")

    chunks = split_documents(documents)
    # print(f"Chunks : {chunks}")
    
    add_to_chroma(chunks)


def load_documents():

    cursor = mysql_conn.cursor()
    cursor.execute("SELECT film_id, title, description FROM film")
    
    docs = []
    for film_id, title, description in cursor.fetchall():
        str = f"Id: {film_id}\nTitle: {title}\nDescription: {description}\n"
        doc = Document(page_content=str, metadata={"mysql_id": film_id, "source":"film"})
        docs.append(doc)

    return docs

def generate_descriptions():

    PROMPT_TEMPLATE = """
    You will generate at least 3 paragraph of a movie description and synopsis.
    You do not need to style the headings.

    {context}

    Generate the response for the movie title: {question}
    """
    model = Ollama(model="llama3.1:8b-instruct-q8_0")

    cursor = mysql_conn.cursor()
    cursor.execute("SELECT film_id, title FROM film WHERE film_id IN (85, 528)")
    
    update_sql = "UPDATE film SET description = %s WHERE film_id = %s"
    for film_id, title in cursor.fetchall():
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context="", question=title)

        response_text = model.invoke(prompt)
        str = f"Title: {title}\nDescription: {response_text}\n"
        print(str)

        try:
            cursor.execute(update_sql, (response_text, film_id))
            mysql_conn.commit()        
        except mysql.connector.Error as err:
            print(f"Error: {err}")
            mysql_conn.rollback()

    cursor.close()
    mysql_conn.close()

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=40,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Calculate chunk_id metadata.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    db.add_documents(chunks_with_ids)


def calculate_chunk_ids(chunks):

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("mysql_id")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["chunk_id"] = chunk_id
        chunk.metadata["chunk_index"] = current_chunk_index

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()
