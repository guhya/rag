import logging
import uuid

from langgraph.graph import END, StateGraph, START

from line.agents.state import State
from line.agents.evaluation_agent import evaluation_agent

from langchain.globals import set_debug

logger = logging.getLogger(__name__)

set_debug(False)


builder = StateGraph(State)
builder.add_node("evaluation_agent", evaluation_agent)


builder.add_edge("__start__", "evaluation_agent")
builder.add_edge("evaluation_agent", "__end__")

app = builder.compile()

def agent_rag(user_prompt: str):
    thread_id = str(uuid.uuid4())
    config = {
        "configurable": {
            "user_id": "anonymous",
            "thread_id": thread_id,
        }
    }

    inputs = {"ori_prompt": user_prompt}
    response = {}
    for output in app.stream(inputs, config, stream_mode="values"):
        response = output

    return response["evaluation"]
