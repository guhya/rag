import logging
from langchain_chroma import Chroma

from utils import ew_embedding_util 
from film.service import film_insert_service
from film.service import film_delete_service

# Global variable start
db = ew_embedding_util.get_chroma_db("chroma_film_indonesia")
logger = logging.getLogger(__name__)
# Global variable ends

def update_item(id, payload):
    """Update an item at a specific id."""

    try:
        id = int(id)
        # Delete existing
        film_delete_service.delete_item(id)

        # Insert afterwards
        logger.debug(f"Payload [{payload}]")
        film_insert_service.insert_item(id, payload)

        logger.debug(f"Item at id [{id}] updated")

    except Exception as e:
        logger.error(f"Update Error: {e}")

