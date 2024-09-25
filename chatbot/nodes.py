import logging

from chatbot.state import State
from chatbot.tools import *

from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)


def get_user_info(state: State):
    return {
        "user_info": {
                "name"      : "Amanda"
                , "age"     : 20
                , "gender"  : "Female"
                , "status"  : "unmarried"
            }
        }
    
    
PREPARE_TOOLS_SYSTEM_PROMPT = """
            You will be an assesor who can assess that a prompt from user required external tool call or not.
               - If you need to use tool call or external resources to answer the prompt, write 'use_tools' and nothing else.
               - If you does not need to use tool call or external resources, write 'general' and nothing else.

            Here are some examples of questions that typically require a tool_call:
            1. Personal data: Questions about specific personal information, such as:
                - What's my age?
                - Where do I live?
                - What's my email address?
            2. Location-based queries: Questions that rely on current or real-time location information, such as:
                - What's the weather like in [city] today?
                - What's the traffic situation in [location] right now?
                - Is it cloudy today?
            3. Real-time data: Questions that require up-to-date or dynamic data, such as:
                - What are the latest sports scores?
                - What's the current stock price of [company]?
            4. External services: Questions that rely on external resources, APIs or services, such as:
                - Can you translate this text from English to Spanish?
                - Can you summarize this article for me?


            On the other hand, here are some examples of questions that typically do not require a Tool Call:

            1. General knowledge: Questions about general information, such as:
                - What's the capital of France?
                - Who wrote the book "To Kill a Mockingbird"?
            2. Fictional or hypothetical scenarios: Questions that involve fictional characters, places, or situations, such as:
                - What would happen if Harry Potter were to defeat Voldemort?
                - How would you write a story about a character who can fly?
            3. Abstract concepts: Questions that explore abstract ideas or philosophical topics, such as:
                - What is the nature of consciousness?
                - Is it possible for humans to live in space?
                
            The prompt : {prompt}
            """

def prepare_tools(state: State):
    logger.debug(f"##### State: \n{state}")
    original_prompt = state["messages"][-1].content
    prompt_llm = ChatOllama(model="llama3.1:8b-instruct-q8_0", temperature = 1)
    prompt_template = ChatPromptTemplate.from_template(PREPARE_TOOLS_SYSTEM_PROMPT)
    prompt = prompt_template.format(prompt=original_prompt)
    result = prompt_llm.invoke(prompt)
    logger.debug(f"##### Prepare tools result: [{result.content}]")
    if "use_tools" != result.content:
        return {"selected_tools": []}    
    else:    
        return {"selected_tools": ["get_school_picture"]}    

def clear_tools(state: State):
    return {"selected_tools": []}    

