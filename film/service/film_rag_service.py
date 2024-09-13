import logging
import mysql.connector

from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from film.io.film_score import FilmScore
from film.service import film_search_service
from film.service import film_keyword_extractor

# Global variable start
mysql_conn = mysql.connector.connect(
    host="localhost",
    user="sakila",
    password="sakila",
    database="sakila"
)

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """
You are a helpful assistant who will judge that movies listed below are related to the user provided keywords or not.
Here is the description of the movies :

{context}

Explain why the description fit the question below:
{question}

Respond only with valid JSON. Do not write an introduction or summary.
Here is an example output: 
[
    {{
        "film_id": 60,
        "score": 0.4038965702056885,
        "title": "FILM TITLE",
        "related" : "No"
        "llm_summary": ""
    }},
]

If you think the movie is not related to the question, write 'No' in 'related' field, otherwise write 'Yes'.
In 'llm_summary' field, write short description why this movie is related to the question above.
"""
# Global variable ends

def film_rag(query_text: str):
    """Add additional context to a keywords to enhance result"""
    
    # Get keywords describing more details of user query
    keywords = film_keyword_extractor.film_get_keywords(query_text)
    logger.info(f"Extracted keywords [{keywords}]")

    chroma_results = film_search_service.film_search(keywords)    
    if len(chroma_results) < 1:
        logger.error("Data not found")

    film_id = chroma_results[0].film_id
    
    # Get additional data from MySQL, add it to the dict
    cursor = mysql_conn.cursor(dictionary=True)
    qry = f"SELECT film_id, title, description FROM film WHERE film_id IN ({film_id})"
    cursor.execute(qry)    
    resultset = cursor.fetchall()
    for rs in resultset:
        chroma_results[0].title = rs["title"]
        chroma_results[0].description = rs["description"]
    cursor.close()

    user_query = query_text + ". " + keywords
    logger.debug(f"User query : {user_query}")
    formatted_response = call_ollama_with(chroma_results[0], user_query)

    return formatted_response


def call_ollama_with(film: FilmScore, query_text: str):
    """Send the prompt to LLM engine"""

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    context_text = film.to_str()
    context_text = context_text.replace("\"", "`")

    prompt = prompt_template.format(context=context_text, question=query_text)
    logger.debug(f"Sending prompt to LLM.. {prompt}")

    # llama3.1:70b-instruct-q2_k
    # llama3.1:8b-instruct-q8_0
    model = Ollama(model="llama3.1:8b-instruct-q8_0",temperature = 0.7)
    response_text = model.invoke(prompt)
    formatted_response = f"Response: {response_text}"

    return formatted_response

def search_mysql(keywords):
    return ""