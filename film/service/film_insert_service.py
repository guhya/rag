import logging

from langchain.schema.document import Document

from utils import ew_embedding_util 
from film.service import film_list_service

# Global variable start
db = ew_embedding_util.get_chroma_db("chroma_film_indonesia")
logger = logging.getLogger(__name__)
# Global variable ends


def insert_item(id, payload):
    """Insert a new item """
    
    try:
        id = int(id)
        # Check for existing item
        existing = film_list_service.list_items([id])
        logger.debug(f"Exist : [{len(existing)}]")
        if len(existing) > 0:
            print(f"Existing item found, please use update operation to update the item")
            return

        logger.debug(f"Payload [{payload}]")
        doc = Document(page_content=payload, metadata={"mysql_id": id, "source":"film"})
        chunks = ew_embedding_util.split_documents([doc])
        ew_embedding_util.add_to_chroma(chunks, db)

        logger.debug(f"Item inserted at id [{id}]")

    except Exception as e:
        logger.error(f"Insert Error: {e}")