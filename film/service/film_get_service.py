import logging

from langchain.schema.document import Document
from langchain_chroma import Chroma

from utils import ew_embedding_util 
from film.service import film_list_service

# Global variable start
db = ew_embedding_util.get_chroma_db("chroma_film_indonesia")
logger = logging.getLogger(__name__)
# Global variable ends


def get_item(id):
    """Get an item by id."""

    try:
        id = int(id)
        logger.debug(f"Get [{id}]")
        existing = film_list_service.list_items([id])
        if len(existing) < 1:
            logger.error("Data not found")
            
        return existing

    except Exception as e:
        logger.error(f"Get Error: {e}")
