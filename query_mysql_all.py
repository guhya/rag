import argparse
import mysql.connector
import json

from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from get_embedding_function import get_embedding_function
from film_score import FilmScore

CHROMA_PATH = "chroma_film"

# Setup MySQL and Chroma DB
mysql_conn = mysql.connector.connect(
    host="localhost",
    user="sakila",
    password="sakila",
    database="sakila"
)


PROMPT_TEMPLATE = """
Summarize each movie below into a paragraph, do not add or remove another content:

{context}

---

Generate your response in JSON, with id included in the JSON object.
"""


PROMPT_TEMPLATE_JSON = """
Respond only with valid JSON. Do not write an introduction or summary.

Here is an example input:
[
    {{
        "film_id": 60,
        "score": 0.4038965702056885,
        "title": "BEAST HUNCHBACK",
        "description": "Description about the movie",
        "llm_summary": null
    }},
    {{
        "film_id": 66,
        "score": 0.39951908588409424,
        "title": "BENEATH RUSH",
        "description": "Description about the movie",
        "llm_summary": null
    }}
]

Here is an example output: 
[
    {{
        "film_id": 60,
        "score": 0.4038965702056885,
        "title": "BEAST HUNCHBACK",
        "description": "Description about the movie",
        "llm_summary": "Summary generated from movie description"
    }},
    {{
        "film_id": 66,
        "score": 0.39951908588409424,
        "title": "BENEATH RUSH",
        "description": "Description about the movie",
        "llm_summary": "Summary generated from movie description"
    }}
]

Here is the real input:

{context}

Do not add or remove item from the list input.
Examine each item and write short summary explaining why this item fits the description of this question : {question}
Put the summary in llm_summary field.

"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("--llm", action='store_true', help="Send prompt to LLM.")
    
    args = parser.parse_args()
    query_text = args.query_text

    query_rag(query_text, args.llm)


def query_rag(query_text: str, send_to_llm: bool):
    # Prepare the DB.
    db = Chroma(persist_directory=CHROMA_PATH
                , embedding_function=get_embedding_function()
                , collection_metadata={"hnsw:space": "cosine"})

    # Search the DB.
    k = 5
    threshold = 0.4
    print(f"Querying vector stores with top [{k}] threshold [{threshold}]: {query_text}")
    results = db.similarity_search_with_score(query_text, k)

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

    # Get additional data from MySQL, add it to the dict
    cursor = mysql_conn.cursor(dictionary=True)
    film_id_str = ",".join(map(str, film_id_set))
    qry = f"SELECT film_id, title, description FROM film WHERE film_id IN ({film_id_str})"
    cursor.execute(qry)    
    resultset = cursor.fetchall()
    for rs in resultset:
        doc_dict[rs["film_id"]].title = rs["title"]
        doc_dict[rs["film_id"]].description = rs["description"]
    cursor.close()

    # Convert dictionary to list and sort again descendingly 
    doc_list_final = sorted(list(doc_dict.values()), key=lambda fs: fs.score, reverse=True)

    formatted_response = f"Sources: {doc_list_final}"

    # Send to LLM if flag is set
    if send_to_llm:
        formatted_response = call_ollama_with_json(doc_list_final, query_text)

    print(formatted_response)
    return formatted_response


def combine_and_sort(docs: list, metadatas: list):
    i = 0
    combined_docs = [];
    for doc in metadatas:
        tmp = metadatas[i]
        tmp["content"] = docs[i]
        i += 1
        combined_docs.append(tmp)
        
    combined_docs_sorted = sorted(combined_docs, key=lambda x: (x['mysql_id'], x['chunk_index']))
    return combined_docs_sorted


def generate_context(combined_docs_sorted: list):
    mysql_id = -1
    context_text = ""
    for doc in combined_docs_sorted:
        if mysql_id != doc["mysql_id"]:
            context_text += "\n\n---\n\n"
        context_text += f"Id = [{doc["mysql_id"]}] {doc["content"]}"
        mysql_id = doc["mysql_id"]
    
    return context_text

def call_ollama_with_text(db, doc_set: set, query_text: str):
    # Define the metadata criteria
    metadata_criteria = {"mysql_id": { "$in" : list(doc_set)}}
    print(f"Get all vectors with all the film_id from vector metadata.. : {metadata_criteria}")
    retriever = db.get(where=metadata_criteria)
    docs = retriever.get("documents")
    metadatas = retriever.get("metadatas")

    combined_docs_sorted = combine_and_sort(docs, metadatas)
    print(f"Combine metadata and content into one list...")

    context_text = generate_context(combined_docs_sorted)
    print(f"Generate a context information from the list...")
    
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(f"Sending prompt to LLM.. {prompt}")

    model = Ollama(model="llama3.1:8b-instruct-q8_0",temperature = 0.0)
    response_text = model.invoke(prompt)
    formatted_response = f"Response: {response_text}\n\nSources: {doc_set}"
    return formatted_response

def call_ollama_with_json(doc_list_final: list, query_text: str):
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_JSON)
    json_array = json.dumps([obj.to_dict() for obj in doc_list_final], indent=4)
    context_text = f"{json_array}"

    prompt = prompt_template.format(context=context_text, question=query_text)
    print(f"Sending prompt to LLM.. {prompt}")

    model = Ollama(model="llama3.1:8b-instruct-q8_0",temperature = 0.0)
    response_text = model.invoke(prompt)
    formatted_response = f"Response: {response_text}"
    return formatted_response


if __name__ == "__main__":
    main()
