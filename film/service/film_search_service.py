import logging

from utils import ew_embedding_util
from utils import ew_mysql_util
from film.io.film_score import FilmScore
from film.service import film_keyword_extractor

# Global variable start
db = ew_embedding_util.get_chroma_db("chroma_film_indonesia")
mysql_conn = ew_mysql_util.get_mysql_conn()
logger = logging.getLogger(__name__)

MYSQL_WEIGHT = 0.7
CHROMA_WEIGHT = 0.3
# Global variable ends


def film_search(query_text: str):
    """
    Semantically search data with the provided keywords.
    Here is how the search algorithm works:

    1. Extract all relevant keywords related to user query
        - Pass user query to LLM to get list of keywords

    2. Search top 10 MySQL using natural language search with the keywords
        - Normalize the score from 0 ~ 1
        - Algorithm : 
            max_mysql_score = max(mysql_scores)
            normalized_mysql_scores = [score / max_mysql_score for score in mysql_scores]

    3. Search top 10 ChromaDB using similarity search with the keywords
        - Perform cosine similarity with the score 0 ~ 1

    4. Combine the result from both search and order by score
        - Set the weight for mysql, msyql_weight: 0.3
        - Set the weight for chroma, chroma_weight: 0.7
        - Calculate weight of all list :
            [(msyql_weight * mysql + chroma_weight * chroma) for mysql, chroma in zip(normalized_mysql_scores, chroma_scores)]
        - Combine and sort

    """

    # Extract keywords
    keywords = extract_keywords(query_text)
    logger.info(f"Extracted keywords:\n{keywords}")

    # Search MySQL
    mysql_results = search_mysql(keywords)
    logger.info(f"MySQL results:\n{mysql_results}")

    # Search Chroma
    chroma_results = search_chroma(keywords)
    logger.info(f"Chroma results:\n{chroma_results}")

    # Combine and sort by score
    # Create a dictionary for quick access to semantic scores
    chroma_results_dict = {item.film_id : item.score for item in chroma_results}
    
    # Combine scores using a set of all unique IDs
    combined_scores = []
    for mysql_item in mysql_results:
        chroma_score = chroma_results_dict.get(mysql_item.film_id, 0)  
        combined_score = ((MYSQL_WEIGHT * mysql_item.score) + (CHROMA_WEIGHT * chroma_score))
        combined_score = FilmScore(mysql_item.film_id, combined_score)
        combined_scores.append(combined_score)

    # Add missing semantic search results
    for chroma_item in chroma_results:
        if chroma_item.film_id not in [item.film_id for item in mysql_results]:
            missing_chroma_item = FilmScore(chroma_item.film_id, chroma_item.score)
            logger.debug(f"Missing chroma item :\n{missing_chroma_item}")
            combined_scores.append(missing_chroma_item)        
    
    # Sort by combined score in descending order
    combined_scores.sort(key=lambda film: film.score, reverse=True)
    logger.info(f"Final ordered results:\n{combined_scores}")

    result = {"keywords": keywords, "results": combined_scores}
    return result


def extract_keywords(query_text: str):
    """Extract keywords from query text"""

    keywords = film_keyword_extractor.film_get_keywords(query_text)
    return keywords


def search_mysql(keywords: str):
    """Search mysql db with natural language search using keywords"""
    
    cursor = mysql_conn.cursor(dictionary=True)
    qry = f"""
            SELECT
                film_id
                , MATCH(title, description) AGAINST('{keywords}') AS score
            FROM film_indonesia
            WHERE MATCH(title, description) AGAINST('{keywords}')
            LIMIT 10
            """

    cursor.execute(qry)    
    resultset = cursor.fetchall()
    
    # Normalize score
    max_mysql_score = max(rs["score"] for rs in resultset)
    logger.debug(f"Max mysql score: [{max_mysql_score}]")

    film_list = list()
    for rs in resultset:
        film_score = FilmScore(rs["film_id"], rs["score"] / max_mysql_score)
        film_list.append(film_score)
    cursor.close()


    return film_list


def search_chroma(keywords):
    """Search Chroma DB with cosine similarity search"""

    # Search the DB.
    k = 10
    logger.debug(f"Querying vector stores with top [{k}] : {keywords}")
    results = db.similarity_search_with_score(keywords, k)

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

    # Convert dictionary to list and sort again descendingly 
    doc_list_final = sorted(list(doc_dict.values()), key=lambda fs: fs.score, reverse=True)    
    return doc_list_final