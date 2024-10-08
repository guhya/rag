import logging
import uuid

from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition

from chatbot.state import State
from chatbot.agent import Assistant
from chatbot.tools import *
from chatbot.nodes import *
from chatbot.primary_agent import primary_llm, primary_prompt
from langchain.globals import set_debug

logger = logging.getLogger(__name__)

set_debug(True)


def should_reask_prompt(state: State) -> Literal["prepare_tools", "__end__"]:
    """Return the next node to execute."""

    is_question_related = state["is_question_related"]
    
    # If question is not related to schoolfinding activity, ask better question
    if "no" == is_question_related:
        return "__end__"
    
    # Otherwise, continue
    return "prepare_tools"


def should_use_tools(state: State) -> Literal["tools", "answer_grader"]:
    """Return the next node to execute."""

    generation = state["generation"]
    
    # If there is no function call, then we grade the answer
    if "tool_calls" != generation:
        return "answer_grader"
    
    # Otherwise if there is, we continue
    return "tools"

builder = StateGraph(State)
builder.add_node("primary_assistant", Assistant(primary_llm, primary_prompt))
builder.add_node("tools", create_tool_node_with_fallback(all_tools))
builder.add_node("fetch_user_info", get_user_info)
builder.add_node("prompt_grader", prompt_grader)
builder.add_node("prepare_tools", prepare_tools)
builder.add_node("answer_grader", answer_grader)

builder.add_edge("__start__", "fetch_user_info")
builder.add_edge("fetch_user_info", "prompt_grader")
builder.add_conditional_edges("prompt_grader", 
    should_reask_prompt
)
builder.add_edge("prepare_tools", "primary_assistant")
builder.add_conditional_edges(
    "primary_assistant",
    should_use_tools
)
builder.add_edge("tools", "primary_assistant")
builder.add_edge("answer_grader", "__end__")
builder.add_edge("prompt_grader", "__end__")

# The checkpointer lets the graph persist its state
# this is a complete memory for the entire graph.
memory = MemorySaver()
app = builder.compile(checkpointer=memory)
thread_id = str(uuid.uuid4())
config = {
    "configurable": {
        "user_id": "3442 587242",
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
