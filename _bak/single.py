
import re
import uuid
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_ollama import ChatOllama
from chatbot.tools import *
from langchain_core.tools import StructuredTool
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from utils.ew_embedding_util import get_embedding_function

def create_tool(company: str) -> dict:
    """Create schema for a placeholder tool."""
    formatted_company = re.sub(r"[^\w\s]", "", company).replace(" ", "_")

    def company_tool(year: int) -> str:
        return f"{company} had revenues of $100 in {year}."

    return StructuredTool.from_function(
        company_tool,
        name=formatted_company,
        description=f"Information about {company}",
    )


s_and_p_500_companies = [  # Abbreviated list for demonstration purposes
    "3M",
    "A.O. Smith",
    "Abbott",
    "Accenture",
    "Advanced Micro Devices",
    "Yum! Brands",
    "Zebra Technologies",
    "Zimmer Biomet",
    "Zoetis",
]

tool_registry = {
    str(uuid.uuid4()): create_tool(company) for company in s_and_p_500_companies
}

tool_documents = [
    Document(
        page_content=tool.description,
        id=id,
        metadata={"tool_name": tool.name},
    )
    for id, tool in tool_registry.items()
]

vector_store = InMemoryVectorStore(embedding=get_embedding_function())
document_ids = vector_store.add_documents(tool_documents)

class State(TypedDict):
    messages: Annotated[list, add_messages]
    selected_tools: list[str]


graph_builder = StateGraph(State)

tools = list(tool_registry.values())
llm = ChatOllama(model="llama3.1:8b-instruct-q8_0", temperature = 1)

def agent(state: State):
    selected_tools = [tool_registry[id] for id in state["selected_tools"]]
    llm_with_tools = llm.bind_tools(selected_tools)
    print(f"Tool bindings: {selected_tools}")
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


def select_tools(state: State):
    last_user_message = state["messages"][-1]
    query = last_user_message.content
    tool_documents = vector_store.similarity_search(query)
    return {"selected_tools": [document.id for document in tool_documents]}


graph_builder.add_node("agent", agent)
graph_builder.add_node("select_tools", select_tools)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "agent",
    tools_condition,
)
graph_builder.add_edge("tools", "agent")
graph_builder.add_edge("select_tools", "agent")
graph_builder.add_edge(START, "select_tools")
graph = graph_builder.compile()

thread_id = str(uuid.uuid4())
config = {
    "configurable": {
        "passenger_id": "3442 587242",
        "thread_id": thread_id,
    }
}

_printed = set()
while True:
    prompt = input("User: ")
    if prompt.lower() in ["quit", "exit", "q"]:
        break

    for event in graph.stream({"messages": ("user", prompt)}, config, stream_mode="values"):
        print_event(event, _printed)
