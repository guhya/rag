import logging

from langchain_chroma import Chroma

from utils import ew_embedding_util 

# Global variable start
db = ew_embedding_util.get_chroma_db("chroma_film")
logger = logging.getLogger(__name__)
# Global variable ends


def list_items(ids):
    """List all items in the data_list."""
    
    metadata_criteria = {"mysql_id": { "$in" : ids}}
    logger.debug(f"Get all vectors with all the film_id from vector metadata.. : {metadata_criteria}")
    retriever = db.get(where=metadata_criteria)
    ids = retriever.get("ids")
    docs = retriever.get("documents")
    metadatas = retriever.get("metadatas")
    i = 0
    combined_docs = []
    for doc in metadatas:
        tmp = metadatas[i]
        tmp["content"] = docs[i]
        tmp["vector_id"] = ids[i]
        i += 1
        combined_docs.append(tmp)
        
    # Sort chuncks by their id and chunk number
    combined_docs_sorted = sorted(combined_docs, key=lambda x: (x["mysql_id"], x["chunk_index"]))
    logger.debug(f"Combined and sorted :\n {combined_docs_sorted}")
    
    return combined_docs_sorted
        