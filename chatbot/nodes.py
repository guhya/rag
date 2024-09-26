import logging
from datetime import date, datetime

from chatbot.state import State
from chatbot.tools import *

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

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
    
    
def prepare_tools(state: State):
    prompt_llm = ChatOllama(model="llama3.1:8b-instruct-q8_0", temperature = 0)
    agent_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You will be an assesor who can assess that a prompt from user required external tool call or not.
                - If you need to use tool call or external resources to answer the prompt, write 'use_tools' and nothing else.
                - If you does not need to use tool call or external resources, write your reasoning.

                Here are some examples of prompt that typically require a tool_call:
                1. Personal data: Prompts about specific personal information, such as:
                    - What's my age?
                    - Where do I live?
                    - What's my email address?
                2. Location-based queries: Prompts that rely on current or real-time location information, such as:
                    - What's the weather like in [city] today?
                    - What's the traffic situation in [location] right now?
                    - Is it cloudy today?
                3. Real-time data: Prompts that require up-to-date or dynamic data, such as:
                    - What are the latest sports scores?
                    - What's the current stock price of [company]?
                    - Find me picture of a teacher with id [2].
                    - Find me picture of Mr. Argus
                4. External services: Prompts that rely on external resources, APIs or services, such as:
                    - Can you translate this text from English to Spanish?
                    - Can you summarize this article for me?


                On the other hand, here are some examples of Prompts that typically do not require a Tool Call:

                1. General knowledge: Prompts about general information, such as:
                    - What's the capital of France?
                    - Who wrote the book "To Kill a Mockingbird"?
                2. Fictional or hypothetical scenarios: Prompts that involve fictional characters, places, or situations, such as:
                    - What would happen if Harry Potter were to defeat Voldemort?
                    - How would you write a story about a character who can fly?
                3. Abstract concepts: Prompts that explore abstract ideas or philosophical topics, such as:
                    - What is the nature of consciousness?
                    - Is it possible for humans to live in space?
                
                Asses only the last message if it needs a tool calls with the provided context.                
                \n\nCurrent user:\n<User>\n{user_info}\n</User>"            
                \nCurrent time: {time}.
                """
            ),
            ("placeholder", "{messages}")
        ]
    )
    agent_prompt = agent_prompt.partial(time=datetime.now())
    prompt_llm = agent_prompt | prompt_llm
    result = prompt_llm.invoke(state)
    logger.debug(f"##### Prepare tools result: [{result}]")
    if "use_tools" != result.content:
        return {"selected_tools": []}    
    else:    
        return {"selected_tools": ["get_school_picture"]}    


def answer_grader(state: State):
    answer_grader_llm = ChatOllama(model="llama3.1:8b-instruct-q8_0", temperature = 0)
    agent_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a grader assessing whether an answer addresses / resolves the last question in the conversation.
                Given the conversation as the context, respond with 'yes' if the answer resolves or addresses the question related to the school and education.
                School finding activity might include greeting and small talk.
                
                Do not write anything else except 'yes' or 'no' in your response.                
                Here are few response examples:
                
                The question: i am hungry
                The answer: I am sorry that you are hungry, however, we cannot help you right now.
                Your Response: yes
                
                The question: Helloo..
                The answer: Hello Ms. Amanda, how are you today? Is there something I can help you with?
                Your Response: yes
                
                The question: Find me picture
                The answer: get_school_picture
                Your Response: no

                The question: I am hungry..
                The answer: Please eat.
                Your Response: no

                Current user: <User>{user_info}</User>"            
                Current time: {time}.
                The question: {question}
                The answer: {generation}.
                The full conversation history:                
                """
            ),
            MessagesPlaceholder(variable_name="messages")
        ]
    )
    agent_prompt = agent_prompt.partial(time=datetime.now())
    question = state["messages"][-1].content
    agent_prompt = agent_prompt.partial(question=question)
    generation = state["generation"]
    agent_prompt = agent_prompt.partial(generation=generation)
    
    answer_grader_llm = agent_prompt | answer_grader_llm
    result = answer_grader_llm.invoke(state)
    logger.debug(f"##### Answer grader result: [{result.content}]")
    if "yes" == result.content:
        return {"messages": AIMessage(content=generation)}
    else:    
        return {"messages": AIMessage(content=f"I am sorry i cannot answer your question. Anything else i can help you with?")}

