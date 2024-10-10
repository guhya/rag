import logging
from film.agents.state import State
from utils import ew_embedding_util
from film.io.film_score import FilmScore

logger = logging.getLogger(__name__)
db = ew_embedding_util.get_chroma_db("chroma_film_indonesia")
MYSQL_WEIGHT = 0.8
CHROMA_WEIGHT = 0.2

def reranking_agent(state: State):
    # Search MySQL
    mysql_results = state["mysql_list"]
    logger.debug(f"MySQL results:\n{mysql_results}")

    # Search Chroma
    chroma_results = state["chroma_list"]
    logger.debug(f"Chroma results:\n{chroma_results}")

    # Combine and sort by score
    # Create a dictionary for quick access to semantic scores
    chroma_results_dict = {item["item_id"] : item["score"] for item in chroma_results}

    # Combine scores using a set of all unique IDs
    logger.info(f"### Combining results MySQL [{len(mysql_results)}] Chroma [{len(chroma_results)}]")
    combined_scores = []
    for mysql_item in mysql_results:
        chroma_score = chroma_results_dict.get(mysql_item["item_id"], 0)  
        combined_score = ((MYSQL_WEIGHT * mysql_item["score"]) + (CHROMA_WEIGHT * chroma_score))
        combined_score = {"item_id":mysql_item["item_id"], "score": combined_score}
        combined_scores.append(combined_score)

    # Add missing semantic search results
    for chroma_item in chroma_results:
        if chroma_item["item_id"] not in [item["item_id"] for item in mysql_results]:
            missing_chroma_item = {"item_id":chroma_item["item_id"], "score": chroma_item["score"]}
            combined_scores.append(missing_chroma_item)        
    
    # Sort by combined score in descending order
    combined_scores.sort(key=lambda item: item["score"], reverse=True)
    logger.debug(f"### Final ordered results:\n{combined_scores}")

    combined_list = list()
    for item in combined_scores:
        item_obj = {"item_id":item["item_id"], "score":item["score"]}
        combined_list.append(item_obj)    
    
    return {
            "combined_list": combined_list,
            }