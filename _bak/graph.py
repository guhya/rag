import logging
import json

from typing import Annotated
from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_core.messages import RemoveMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import ToolMessage
from langchain.globals import set_debug
from langchain_core.prompts import ChatPromptTemplate
from datetime import date, datetime

logger = logging.getLogger(__name__)
memory = MemorySaver()


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    # Note : This is only in the context of one set of question and answer
    messages: Annotated[list, add_messages]

    # We will add a `summary` attribute (in addition to `messages` key,
    # which MessagesState already has)
    summary: str



def print_state(graph: StateGraph, config: any):
    for state in graph.get_state_history(config):
        no = len(state.values["messages"])
        logger.debug(f"Num Messages: {no} Next: {state.next}")


def call_llm(state: State):
    response = llm_with_tools.invoke(state["messages"])
                
    return {"messages": [response]}    

@tool
def query_school_database(id: str):
    """
    Query mysql database to get the picture based on id

    Args:
        id: The id of the school

    Returns:
        The path of the pictures of that particular id

    """

    return []

@tool
def nuke_the_world(query: str):
    """
    This function will nuke the world

    Args:
        query: The launch code

    Returns:
        Void, everything is destroyed

    """

    return []

# mistral:latest
# llama3.1:8b-instruct-q8_0
llm = ChatOllama(model="llama3.1:8b-instruct-q8_0", temperature = 1)
primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful customer support assistant for Schoolfinder Inc."
            "Use the provided tools to search for schoools and other information to assist the user's queries."
            "Do not use provided tools when answering normal question not related to the school finding activities"
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())
tools = [nuke_the_world, query_school_database]
llm_with_tools = llm.bind_tools(tools)

def should_continue(state: State) -> Literal["tools", "__end__"]:
    """Return the next node to execute."""

    last_message = state["messages"][-1]
    logger.info(f"Tool :{last_message.tool_calls}")
    
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "__end__"
    
    # Otherwise if there is, we continue
    return "tools"


def summarize_conversation(state: State):
    # First, we summarize the conversation
    summary = state.get("summary", "")
    if summary:
        # If a summary already exists, we use a different system prompt
        # to summarize it than if one didn't
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"

    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = llm.invoke(messages)

    # We now need to delete messages that we no longer want to show up
    # I will delete all but the last two messages, but you can change this
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}


tool_node = ToolNode(tools)
graph_builder = StateGraph(State)
# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("call_llm", call_llm)
graph_builder.add_node("tools", tool_node)

# Set the entrypoint as 'call_llm'
# This means that this node is the first one called
graph_builder.add_edge(START, "call_llm")

# We now add a conditional edge
graph_builder.add_conditional_edges(
    # First, we define the start node. We use `call_llm`.
    # This means these are the edges taken after the `agent` node is called.
    "call_llm",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
)
graph_builder.add_edge("tools", "call_llm")

app = graph_builder.compile(checkpointer=memory, interrupt_before=[])

config = {"configurable": {"thread_id": "1"}}

while True:

    prompt = input("User: ")
    if prompt.lower() in ["quit", "exit", "q"]:
        logger.info("Goodbye!")
        break

    user_input = HumanMessage(content=prompt)
    for event in app.stream({"messages": [user_input]}, config):
        print_state(app, config)
        list_response = list(event.values())
        response = list_response[-1]["messages"]
        if len(response) > 0:
            for value in event.values():
                logger.info(f"Assistant:\n{response[-1].content}")