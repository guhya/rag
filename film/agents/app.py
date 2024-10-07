import logging
import uuid

from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition

from film.agents.state import State
from film.agents.keyword_agent import keyword_agent
from film.agents.mysql_agent import mysql_agent
from film.agents.chroma_agent import chroma_agent
from film.agents.reranking_agent import reranking_agent
from film.agents.evaluation_agent import evaluation_agent
from film.agents.utils import *

from langchain.globals import set_debug

logger = logging.getLogger(__name__)

set_debug(False)

"""
Semantically search data with the provided keywords.
Here is how the search algorithm works:

1. Extract all relevant keywords related to user query
    - Pass user query to LLM to get list of keywords

2. Search top 10 MySQL using natural language search with the keywords
    - Normalize the score from 0 ~ 1
    - Algorithm : 
        max_mysql_score = max(mysql_scores)
        normalized_mysql_scores = [score / max_mysql_score for score in mysql_scores]

3. Search top 10 ChromaDB using similarity search with the keywords
    - Perform cosine similarity with the score 0 ~ 1

4. Combine the result from both search and order by score
    - Set the weight for mysql, msyql_weight: 0.3
    - Set the weight for chroma, chroma_weight: 0.7
    - Calculate weight of all list :
        [(msyql_weight * mysql + chroma_weight * chroma) for mysql, chroma in zip(normalized_mysql_scores, chroma_scores)]
    - Combine and sort
"""

builder = StateGraph(State)
builder.add_node("keyword_agent", keyword_agent)
builder.add_node("mysql_agent", mysql_agent)
builder.add_node("chroma_agent", chroma_agent)
builder.add_node("reranking_agent", reranking_agent)
builder.add_node("evaluation_agent", evaluation_agent)


builder.add_edge("__start__", "keyword_agent")
builder.add_edge("keyword_agent", "mysql_agent")
builder.add_edge("mysql_agent", "chroma_agent")
builder.add_edge("chroma_agent", "reranking_agent")
builder.add_edge("reranking_agent", "evaluation_agent")
builder.add_edge("evaluation_agent", "__end__")

# The checkpointer lets the graph persist its state
# this is a complete memory for the entire graph.
memory = MemorySaver()
app = builder.compile(checkpointer=memory)
thread_id = str(uuid.uuid4())
config = {
    "configurable": {
        "user_id": "anonymous",
        "thread_id": thread_id,
    }
}


_printed = set()
while True:
    prompt = input("User: ")
    if prompt.lower() in ["quit", "exit", "q"]:
        logger.info("Goodbye!")
        break

    for event in app.stream({"messages": ("user", prompt)}, config, stream_mode="values"):
        print_event(event, _printed)
