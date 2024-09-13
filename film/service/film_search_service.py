import logging
import mysql.connector

from langchain_chroma import Chroma

from utils import ew_embedding_util
from film.io.film_score import FilmScore

# Global variable start
db = ew_embedding_util.get_chroma_db("chroma_film")
logger = logging.getLogger(__name__)
mysql_conn = mysql.connector.connect(
    host="localhost",
    user="sakila",
    password="sakila",
    database="sakila"
)

# Global variable ends

def film_search(query_text: str):
    """Semantically search data with the provided keywords"""

    # Search the DB.
    k = 20
    logger.debug(f"Querying vector stores with top [{k}] : {query_text}")
    results = db.similarity_search_with_score(query_text, k)

    # Create a list of result
    doc_list = list()
    film_id_set = set()
    for doc, _score in results:
        film_score = FilmScore(doc.metadata.get("mysql_id", None), _score)
        doc_list.append(film_score)
        film_id_set.add(film_score.film_id)

    # Sort the list at its score descendingly, 
    # to make sure that the duplicate film_id with the greatest score will be picked up
    doc_list_desc = sorted(doc_list, key=lambda fs: fs.score, reverse=True)

    # Get unique item only
    doc_set = set()
    for item in doc_list_desc:
        doc_set.add(item)

    # Convert to dict
    doc_dict = dict()
    for item in doc_set:
        doc_dict[item.film_id] = item

    # Get additional data from MySQL, add it to the dict
    cursor = mysql_conn.cursor(dictionary=True)
    film_id_str = ",".join(map(str, film_id_set))
    qry = f"SELECT film_id, title, description FROM film WHERE film_id IN ({film_id_str})"
    cursor.execute(qry)    
    resultset = cursor.fetchall()
    for rs in resultset:
        doc_dict[rs["film_id"]].title = rs["title"]
        doc_dict[rs["film_id"]].description = rs["description"]
    cursor.close()

    # Convert dictionary to list and sort again descendingly 
    doc_list_final = sorted(list(doc_dict.values()), key=lambda fs: fs.score, reverse=True)
    
    return doc_list_final


def combine_and_sort(docs: list, metadatas: list):
    """Combine the similarity search result into one unified list and reverse the sort order"""

    i = 0
    combined_docs = [];
    for doc in metadatas:
        tmp = metadatas[i]
        tmp["content"] = docs[i]
        i += 1
        combined_docs.append(tmp)
        
    combined_docs_sorted = sorted(combined_docs, key=lambda x: (x["mysql_id"], x["chunk_index"]))

    return combined_docs_sorted
