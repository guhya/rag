import logging
import json
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
    
def prompt_grader(state: State):
    # llama3.2:3b-instruct-q8_0
    # llama3.1:8b-instruct-q8_0
    prompt_llm = ChatOllama(model="llama3.2:3b-instruct-q8_0", temperature = 0, format="json")
    agent_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You will be an assesor who can assess that a prompt from user is related to the school or education topic.
                You will not allow a question which falls beyond school or education topic.
                Small talks and greeting is allowed and therefore falls within school or education topic.
                Negative topic is not allowed.
                    
                Score:
                A score of yes means that the prompt is within the topic. 
                A score of no means that the prompt is outside the topic.
                Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct.
                Avoid simply stating the correct answer at the outset." 
                
                Current user:<User>{user_info}</User>"            
                Current time: {time}.
                """
            ),
            (            
                "human", 
                """
                Asses this prompt: {user_prompt}
                Return JSON with three keys, 
                binary_score is 'yes' or 'no' score to indicate that the prompt is related to the school or education topic. 
                A key, explanation, that contains an explanation of the score.
                Another key, additional_message, a kind response from you to ask user to provide with better question ONLY when the binary_score you give is 'no'.                
                """
            )
        ]
    )
    question = state["messages"][-1].content
    logger.debug(f"### User prompt: [{question}]")
    agent_prompt = agent_prompt.partial(time=datetime.now())
    agent_prompt = agent_prompt.partial(user_prompt=question)
    prompt_llm = agent_prompt | prompt_llm
    result = prompt_llm.invoke(state)
    logger.debug(f"##### Prompt grader result: [{result}]")
    ai_response = json.loads(result.content)["binary_score"]
    if "yes" == ai_response:
        return {"is_question_related": ai_response, "question": question}    
    else:    
        additional_message = json.loads(result.content)["additional_message"]
        return {"is_question_related": ai_response, "messages": AIMessage(content=f"{additional_message}")}

    
def prepare_tools(state: State):
    # llama3.2:3b-instruct-q8_0
    # llama3.1:8b-instruct-q8_0
    prompt_llm = ChatOllama(model="llama3.2:3b-instruct-q8_0", temperature = 0, format="json")
    agent_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You will be an assesor who can assess that a prompt from user requires you to use external tool call or not.
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
                    
                Score:
                A score of yes means that the prompt requires external tool calls to reply. 
                A score of no means that the prompt does not require external tool calls to reply.
                Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct.
                Avoid simply stating the correct answer at the outset." 
                
                Current user:<User>{user_info}</User>"            
                Current time: {time}.
                """
            ),
            (            
                "human", 
                """
                Asses this prompt: {user_prompt}
                Return JSON with two two keys, 
                binary_score is 'yes' or 'no' score to indicate that the prompt requires external tool calls. 
                And a key, explanation, that contains an explanation of the score.                
                """
            )
        ]
    )
    question = state["messages"][-1].content
    logger.debug(f"### User prompt: [{question}]")
    agent_prompt = agent_prompt.partial(time=datetime.now())
    agent_prompt = agent_prompt.partial(user_prompt=question)
    prompt_llm = agent_prompt | prompt_llm
    result = prompt_llm.invoke(state)
    logger.debug(f"##### Prepare tools result: [{result}]")
    ai_response = json.loads(result.content)["binary_score"]
    if "yes" != ai_response:
        return {"question": question, "selected_tools": []}    
    else:    
        tool_list = [key for key in tool_registry.keys()]
        logger.debug(f"{tool_list}")
        return {"question": question, "selected_tools": tool_list}    


def answer_grader(state: State):
    # llama3.2:3b-instruct-q8_0
    # llama3.1:8b-instruct-q8_0
    answer_grader_llm = ChatOllama(model="llama3.2:3b-instruct-q8_0", temperature = 0, format="json")
    agent_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a grader assessing whether an answer addresses / resolves the question.
                
                Score:
                A score of yes means that the answer address the question. 
                A score of no means that the answer does not address the question.
                Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct.
                Avoid simply stating the correct answer at the outset." 
                Current user: <User>{user_info}</User>"            
                Current time: {time}.                
                """
            ),
            (
                "human", 
                """
                The question: {question} 
                The answer: {generation}
                \n
                Return JSON with two two keys, 
                binary_score is 'yes' or 'no' score to indicate that the answer address the question. 
                And a key, explanation, that contains an explanation of the score.                
                """
            )
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
    ai_response = json.loads(result.content)["binary_score"]
    if "yes" == ai_response:
        return {"messages": AIMessage(content=generation)}
    else:    
        return {"messages": AIMessage(content=f"I am sorry i cannot answer your question. Anything else i can help you with?")}

