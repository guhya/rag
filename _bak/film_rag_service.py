import argparse
import mysql.connector
import json

from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from utils import ew_embedding_util
from film.io.film_score import FilmScore

CHROMA_PATH = "chroma_film"

# Setup MySQL and Chroma DB
mysql_conn = mysql.connector.connect(
    host="localhost",
    user="sakila",
    password="sakila",
    database="sakila"
)


PROMPT_TEMPLATE = """
Provide the response based only on the following context:
{context}

Explain why the description fit the question below:
{question}

---

Respond only with valid JSON. Do not write an introduction or summary.
Do not change the value of other fields.
Only change the value of llm_summary field with the list of token found in the description which are related to the question, and describe them in short sentence.
Separate each token by comma.
Here is an example output: 
{{
    "film_id": 60,
    "score": 0.4038965702056885,
    "title": "BEAST HUNCHBACK",
    "llm_summary": ""
}},
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
                , embedding_function=ew_embedding_util.get_embedding_function()
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
        formatted_response = call_ollama_with(doc_list_final[0], query_text)

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

def call_ollama_with(film: FilmScore, query_text: str):
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    context_text = film.to_str()
    context_text = context_text.replace("\"", "`")

    prompt = prompt_template.format(context=context_text, question=query_text)
    print(f"Sending prompt to LLM.. {prompt}")

    # llama3.1:70b-instruct-q2_k
    # llama3.1:8b-instruct-q8_0
    model = Ollama(model="llama3.1:8b-instruct-q8_0",temperature = 0.7)
    response_text = model.invoke(prompt)
    formatted_response = f"Response: {response_text}"
    return formatted_response


if __name__ == "__main__":
    main()
