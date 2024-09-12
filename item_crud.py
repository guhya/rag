import argparse
import os
import shutil

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma_film"

# Global variable start
db = Chroma(persist_directory=CHROMA_PATH
            , embedding_function=get_embedding_function()
            , collection_metadata={"hnsw:space": "cosine"})
# Global variable ends

# Private functions start
def _split_documents(documents: list[Document]):
    """Split document into multiple smaller chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=40,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def _generate_chunk_metadata(chunks):
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

def _add_to_chroma(chunks: list[Document]):
    """Add to chroma """
    # Generate chunk metadata.
    chunks_with_metadata = _generate_chunk_metadata(chunks)
    db.add_documents(chunks_with_metadata)

def _validate_list(values):
    """Validate each value in the list."""
    validated_values = []
    max = 100000
    for value in values:
        try:
            # Convert the value to an integer
            number = int(value)
            # Check if the number is within a specific range (e.g., 0 to 100000)
            if 0 <= number <= max:
                validated_values.append(number)
            else:
                print(f"Value '{value}' is out of range (0-{max}).")
        except ValueError:
            print(f"Value '{value}' is not a valid integer.")
    
    return validated_values

# Private functions end

def list_items(ids):
    """List all items in the data_list."""
    
    metadata_criteria = {"mysql_id": { "$in" : ids}}
    print(f"Get all vectors with all the film_id from vector metadata.. : {metadata_criteria}")
    retriever = db.get(where=metadata_criteria)
    ids = retriever.get("ids")
    docs = retriever.get("documents")
    metadatas = retriever.get("metadatas")
    i = 0
    combined_docs = [];
    for doc in metadatas:
        tmp = metadatas[i]
        tmp["content"] = docs[i]
        tmp["vector_id"] = ids[i]
        i += 1
        combined_docs.append(tmp)
        
    # Sort chuncks by their id and chunk number
    combined_docs_sorted = sorted(combined_docs, key=lambda x: (x["mysql_id"], x["chunk_index"]))
    print(f"List :\n {combined_docs_sorted}")
    return combined_docs_sorted
        

def get_item(id):
    """Get an item by id."""
    try:
        id = int(id)
        print(f"Get [{id}]")
        existing = list_items([id])
        return existing

    except Exception as e:
        print(f"Get Error: {e}")


def update_item(id, payload):
    """Update an item at a specific id."""
    try:
        id = int(id)
        # Delete existing
        delete_item(id)

        # Insert afterwards
        print(f"Payload [{payload}]")
        insert_item(id, payload)

        print(f"Item at id [{id}] updated")

    except Exception as e:
        print(f"Update Error: {e}")


def delete_item(id):
    """Delete an item by id."""
    try:
        id = int(id)
        # Check for existing item
        existing = get_item(id)
        if len(existing) < 1:
            print(f"No existing item found")
            return

        vector_ids = list()
        for chunk in existing:
            vector_ids.append(chunk["vector_id"])

        db.delete(vector_ids)
        print(f"Item at id [{id}] with chunk ids {vector_ids} deleted")

    except Exception as e:
        print(f"Delete Error: {e}")


def insert_item(id, payload):
    """Insert a new item """
    try:
        id = int(id)
        # Check for existing item
        existing = get_item(id)
        print(f"Exist : [{len(existing)}]")
        if len(existing) > 0:
            print(f"Existing item found, please use update operation to update the item")
            return

        print(f"Payload [{payload}]")
        doc = Document(page_content=payload, metadata={"mysql_id": id, "source":"film"})
        chunks = _split_documents([doc])
        _add_to_chroma(chunks)

        print(f"Item inserted at id [{id}]")

    except Exception as e:
        print(f"Insert Error: {e}")


def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Manage a vector db with CRUD operations")

    # Define command-line arguments
    parser.add_argument("--operation", choices=["list", "get", "update", "delete", "insert"], required=True, help="Operation to perform")
    parser.add_argument("--ids", type=str, help="List of comma separated id's to list, only necessary for 'list' operation")
    parser.add_argument("--id", type=str, help="ID or index for the operation")
    parser.add_argument("--payload", type=str, help="Payload for the operation")

    # Parse arguments
    args = parser.parse_args()

    # Execute command based on parsed arguments
    if args.operation == "list":
        if args.ids:
            # Split the comma-separated list into a Python list
            values = [value.strip() for value in args.ids.split(",")]
            # Validate the list
            validated_values = _validate_list(values)
            list_items(validated_values)
        else:
            print("No list provided")        

    elif args.operation == "get":
        if args.id is None:
            parser.error("The 'get' operation requires an --id argument")
        get_item(args.id)

    elif args.operation == "update":
        if args.id is None or args.payload is None:
            parser.error("The 'update' operation requires both --id and --payload arguments")
        update_item(args.id, args.payload)

    elif args.operation == "delete":
        if args.id is None:
            parser.error("The 'delete' operation requires an --id argument")
        delete_item(args.id)

    elif args.operation == "insert":
        if args.id is None or args.payload is None:
            parser.error("The 'insert' operation requires both --id and --payload arguments")
        insert_item(args.id, args.payload)

if __name__ == "__main__":
    main()


