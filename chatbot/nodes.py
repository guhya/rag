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
                Schoolfinder is an organization that helps user finding their dream school for them or for their kids.
                School includes a pre-school, elementary school, middle school, high school, university, islamic based school.
                It offers school search and school recommendation services located only in Indonesia.
                It also offers general consultation about finding good schools in general.
                It has directory of thousands Indonesian schools registered in its system.
                
                Given full conversation as a context, you will be an assesor who can assess that the last prompt from user is related to Schoolfinder services.
                You will not allow a question which falls beyond Schoolfinder services.
                Small talks and greeting is allowed and therefore falls within Schoolfinder services.
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
                Asses this prompt: {user_prompt}. based on following context: {conversation}
                Return JSON with three keys, 
                binary_score is 'yes' or 'no' score to indicate that the last prompt is related to the Schoolfinder services. 
                A key, explanation, that contains an explanation of the score.
                Another key, additional_message, a kind response from you to ask user to provide with better question ONLY when the binary_score you give is 'no'.
                
                Reminder:
                Asses only the last prompt, not the whole conversation.
                """
            )
        ]
    )    
    question = state["messages"][-1].content
    logger.debug(f"### User prompt: [{question}]")
    agent_prompt = agent_prompt.partial(time=datetime.now())
    
    conversation = ""
    role = ""
    for msg in state["messages"]:
        if type(msg) is HumanMessage:
            role = "Human: "
        elif type(msg) is ToolMessage:
            role = "Tool: "
        else:
            role = "AI Response: "
        conversation += role + msg.content + "\n"
    logger.debug(f"Full Conversation:\n{conversation}")
        
    agent_prompt = agent_prompt.partial(user_prompt=question, conversation=conversation)
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
    answer_grader_llm = ChatOllama(model="llama3.1:8b-instruct-q8_0", temperature = 0, format="json")
    agent_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a grader assessing whether an answer addresses / resolves the question.
                The question and the answer might be in Indonesian language.
                
                Score:
                A score of yes means that the answer address the question. 
                A score of no means that the answer does not address the question.
                Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct.
                Avoid simply stating the correct answer at the outset." 
                If the question and answer is within the category of greeting, then asses as yes.
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

