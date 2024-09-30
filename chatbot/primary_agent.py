from datetime import date, datetime

from langgraph.prebuilt import tools_condition
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from chatbot.tools import *
from chatbot.nodes import *

# llama3.1:70b-instruct-q2_k
# llama3.1:8b-instruct-q8_0
# llama3.2:3b-instruct-q8_0
primary_llm = ChatOllama(model="llama3.2:3b-instruct-q8_0", temperature = 1)
primary_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful AI customer support assistant for Schoolfinder Inc.            
            You will be given tools to choose from should you need it to complete the answer.
            Be kind when answering.
            
            Reminder:
            - Do not use tools that are not suitable to answer the question.
            - When you get the result from using the tools, do not modify the result.
            - Do not provide any information about what tools you use to answer the prompt.
            
            Current user:<User>{user_info}</User>"            
            Current time: {time}.
            """
        ),
        ("placeholder", "{messages}")
    ]
)

primary_prompt = primary_prompt.partial(time=datetime.now())
