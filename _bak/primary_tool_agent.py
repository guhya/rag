from datetime import date, datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from chatbot.tools import *
from chatbot.nodes import *

# llama3.1:70b-instruct-q2_k
# llama3.1:8b-instruct-q8_0
llm_with_tool = ChatOllama(model="llama3.1:70b-instruct-q2_k", temperature = 1)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful customer support assistant for Schoolfinder Inc.
            Use proper tools assigned to you when necessarry. 
            If you cannot find suitable tools from the provided tool list to use, do not use any tools and say you cannot fulfill the request.
            \n\nCurrent user:\n<User>\n{user_info}\n</User>"            
            \nCurrent time: {time}.
            """
        ),
        ("placeholder", "{messages}")
    ]
)

primary_tools = [
    get_school_picture
]

prompt = prompt.partial(time=datetime.now())
primary_assistant_with_tool = prompt | llm_with_tool.bind_tools(primary_tools)