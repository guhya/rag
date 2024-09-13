import logging

from utils import ew_embedding_util 
from film.service import film_get_service

# Global variable start
db = ew_embedding_util.get_chroma_db("chroma_film")
logger = logging.getLogger(__name__)
# Global variable ends

def delete_item(id):
    """Delete an item by id."""

    try:
        id = int(id)
        # Check for existing item
        existing = film_get_service.get_item(id)
        if len(existing) < 1:
            logger.error(f"No existing item found")
            return

        vector_ids = list()
        for chunk in existing:
            vector_ids.append(chunk["vector_id"])

        db.delete(vector_ids)
        logger.debug(f"Item at id [{id}] with chunk ids {vector_ids} deleted")

    except Exception as e:
        logger.error(f"Delete Error: {e}")
