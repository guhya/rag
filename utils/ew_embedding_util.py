import logging
import os
import shutil

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain.schema.document import Document
from langchain_chroma import Chroma
from chromadb.config import Settings

logger = logging.getLogger(__name__)


def get_chroma_db(path):
    """Get DB"""
    client_settings = Settings(
        is_persistent=True,
        persist_directory=path,
        anonymized_telemetry=False,
    )

    db = Chroma(embedding_function=get_embedding_function()
            , collection_metadata=get_embedding_similarity()
            , client_settings = client_settings)
    
    return db

def clear_chroma_db(path):
    if os.path.exists(path):
        shutil.rmtree(path)


def split_documents(documents: list[Document]):
    """Split document into multiple smaller chunks"""

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=40,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def generate_chunk_metadata(chunks):
    """Determine the metadata of the chunk"""

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

def add_to_chroma(chunks: list[Document], db: Chroma): 
    """Add to chroma """

    # Generate chunk metadata.
    chunks_with_metadata = generate_chunk_metadata(chunks)
    db.add_documents(chunks_with_metadata)


def get_embedding_function():
    """Use offline embedding provided by Ollama"""
    
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    return embeddings

def get_embedding_similarity():
    return {"hnsw:space": "cosine"}
