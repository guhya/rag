import logging
import mysql.connector

from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from film.io.film_score import FilmScore
from film.service import film_search_service

# Global variable start
logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """
I am running a movie database service, a user enter this in a search query: {question}

Give me set of 10 keywords, comma separated,  which might be related to above question.
Here is an example output: movies, beautiful, singer, music, performance, talent, vocalist, film, artist, musical
Do not write an introduction or summary in your response.
"""
# Global variable ends

def film_get_keywords(query_text: str):
    """Keyword extractors"""
    
    formatted_response = call_ollama_with(query_text)

    return formatted_response


def call_ollama_with(query_text: str):
    """Send the prompt to LLM engine"""

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(question=query_text)
    logger.debug(f"Sending prompt to LLM.. {prompt}")

    # llama3.1:70b-instruct-q2_k
    # llama3.1:8b-instruct-q8_0
    model = Ollama(model="llama3.1:8b-instruct-q8_0",temperature = 0.0)
    response_text = model.invoke(prompt)
    formatted_response = f"{response_text}"

    return formatted_response
