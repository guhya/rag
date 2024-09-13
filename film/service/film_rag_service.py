import logging

from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from film.io.film_score import FilmScore
from film.service import film_search_service

# Global variable start
logger = logging.getLogger(__name__)

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
    "title": "FILM TITLE",
    "llm_summary": ""
}},
"""
# Global variable ends

def film_rag(query_text: str):
    """Add additional context to a keywords to enhance result"""
    
    results = film_search_service.film_search(query_text)
    logger.debug(f"Param : {results[0]}")
    formatted_response = call_ollama_with(results[0], query_text)

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
