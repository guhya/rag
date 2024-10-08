from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph.message import AnyMessage, add_messages
from film.io.film_score import FilmScore

class State(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        ori_prompt: Original prompt fromt the user
        ind_prompt: Translated prompt in Indonesian language
        keywords: Comma separated keywords generated by AI
        mysql_list: List of item from MySQL natural language search
        chroma_list: List of item from Chroma vector search
        combined_list: Combined list ordered by their relevance
        result_list: Final list presented to the user
    """
    
    ori_prompt: str
    ind_prompt: str
    keywords: str
    mysql_list: list[any]
    chroma_list: list[any]
    combined_list: list[any]        
    result_list: list[any]        