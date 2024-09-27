from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda

from langgraph.prebuilt import ToolNode

# Tools must accept String and return String
# Each tools must include a docstring describing its purpose

@tool
def get_school_picture(school_id: str):
    """
    Query mysql database to get the picture based on id

    Args:
        school_id: An integer id of the school

    Returns:
        The path of the pictures of that particular id

    """

    path = f"~/Downloads/school_picture_{school_id}_2024.jpg"
    response = f"The school picture is saved in: {path}"
    return response

@tool
def delete_school_data(school_id: str):
    """
    Esecute mysql script to delete school information based on school id

    Args:
        school_id: An integer id of the school

    Returns:
        The deleted id

    """

    response = f"The school with id [{school_id}] is deleted."
    return response

@tool
def get_weather_data(city: str):
    """
    Call API to get weather data for the input city

    Args:
        city: City name

    Returns:
        Weather information
    """

    response = f"Weather in {city} is sunny with a strong breeze"
    return response


tool_registry = {}
tool_registry["get_school_picture"] = get_school_picture
tool_registry["delete_school_data"] = delete_school_data
tool_registry["get_weather_data"] = get_weather_data
all_tools = [value for value in tool_registry.values()]

def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


def print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)