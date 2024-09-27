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
    # llama3.2:3b-instruct-q8_0
    # llama3.1:8b-instruct-q8_0
    prompt_llm = ChatOllama(model="llama3.2:3b-instruct-q8_0", temperature = 0)
    agent_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You will be an assesor who can assess that a prompt from user required external tool call or not.
                - If you need to use tool call or external resources to answer the prompt, write 'use_tools' and nothing else.
                - If you does not need to use tool call or external resources, write your reasoning.

                Here are some examples of prompt that typically require a tool_call:
                    - What's my age?
                    - What's the traffic situation in [location] right now?
                    - What's the current stock price of [company]?
                    - Find me picture of a teacher with id [2].
                    - Find me picture of Mr. Argus

                On the other hand, here are some examples of Prompts that typically do not require a Tool Call:
                    - What's the capital of France?
                    - How would you write a story about a character who can fly?
                    - Is it possible for humans to live in space?
                
                \n\nCurrent user:\n<User>\n{user_info}\n</User>"            
                \nCurrent time: {time}.
                """
            ),
            ("human", "Asses this prompt: \n\n {user_prompt}")
        ]
    )
    question = state["messages"][-1].content
    logger.debug(f"### User prompt: [{question}]")
    agent_prompt = agent_prompt.partial(time=datetime.now())
    agent_prompt = agent_prompt.partial(user_prompt=question)
    prompt_llm = agent_prompt | prompt_llm
    result = prompt_llm.invoke(state)
    logger.debug(f"##### Prepare tools result: [{result}]")
    if "use_tools" != result.content:
        return {"question": question, "selected_tools": []}    
    else:    
        tool_list = [key for key in tool_registry.keys()]
        logger.debug(f"{tool_list}")
        return {"question": question, "selected_tools": tool_list}    


def answer_grader(state: State):
    # llama3.2:3b-instruct-q8_0
    # llama3.1:8b-instruct-q8_0
    answer_grader_llm = ChatOllama(model="llama3.2:3b-instruct-q8_0", temperature = 0)
    agent_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a grader assessing whether an answer addresses / resolves the last question in the conversation.
                Given the conversation as the context, respond with binary ['yes', 'no']. 'yes' if the answer resolves or addresses the question related to the school and education.
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
                """
            ),
            ("human", "The question: \n\n {question} \n\n The answer: {generation}")
        ]
    )
    agent_prompt = agent_prompt.partial(time=datetime.now())
    question = state["question"]
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

