import logging
from film.agents.state import State
from utils import ew_embedding_util
from film.io.film_score import FilmScore

logger = logging.getLogger(__name__)
db = ew_embedding_util.get_chroma_db("chroma_film_indonesia")

def chroma_agent(state: State):
    ind_prompt = state["ind_prompt"]
    keywords =  state["keywords"] + " " + ind_prompt
    
    # Search the DB.
    k = 10
    logger.debug(f"### Querying vector stores with top [{k}] : {keywords}")
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
    
    chroma_list = list()
    for item in doc_list_final:
        item_obj = {"item_id":item.film_id, "score":item.score}
        chroma_list.append(item_obj)    
    
    return {
            "chroma_list": chroma_list,
            }