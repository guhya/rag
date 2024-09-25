from datetime import date, datetime

from langgraph.prebuilt import tools_condition
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from chatbot.tools import *
from chatbot.nodes import *

# llama3.1:70b-instruct-q2_k
# llama3.1:8b-instruct-q8_0
primary_llm = ChatOllama(model="llama3.1:8b-instruct-q8_0", temperature = 1)
primary_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful customer support assistant for Schoolfinder Inc.
            Greet the user when replying for first time
            Append proper honorific when addressing user, for example : Hello Ms. Phyllis.., Hello Mr. Johny
            Do not call user directly by name without proper honorific. for example : Hello Johny.
            You will be given tools to choose from should you need it to complete the answer.
            If the tools is not suitable with the prompt, do not use it.
            If you do not have tools to choose from, do not use or generate new tool.
            \n\nCurrent user:\n<User>\n{user_info}\n</User>"            
            \nCurrent time: {time}.
            """
        ),
        ("placeholder", "{messages}")
    ]
)

primary_prompt = primary_prompt.partial(time=datetime.now())
