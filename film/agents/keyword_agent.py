import logging
import json
from datetime import date, datetime

from film.agents.state import State

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

logger = logging.getLogger(__name__)

    
def keyword_agent(state: State):
    # llama3.2:3b-instruct-q8_0
    # llama3.1:8b-instruct-q8_0
    prompt_llm = ChatOllama(model="llama3.1:8b-instruct-q8_0", temperature = 0.5, format="json")
    agent_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                I am running a movie database service in Indonesian language where user can search movie based on prompt they entered.
                You will be a prompt assistant whose job is to generate related keywords within the context of user prompt.
                Take your time to understand the prompt properly before generating keywords.
                Take into account whether the prompt is asking about overall movie theme or just part of it.
                The keywords is comma separated text, which must be related to the prompt.                

                Reminder:
                - The keywords should not make the context broader.
                - The keywords should not contains 'movie' or 'film'.
                - Make sure all keywords are in Indonesian language.            
                - Return in plain text like for example: cantik, penyanyi, musik, pertunjukan
                
                Current time: {time}.
                """
            ),
            (            
                "human", 
                """
                Given the prompt: {ori_prompt}, generate comma separated keywords very closely related with that context.                
                Return JSON with two two keys, 
                keywords: keywords extracted. 
                And a key, translation, user prompt translated into Indonesian language if necessary, if not just the original prompt.                
                """
            )
        ]
    )    
    ori_prompt = state["ori_prompt"]
    logger.debug(f"### User prompt: [{ori_prompt}]")
    agent_prompt = agent_prompt.partial(time=datetime.now())    
    agent_prompt = agent_prompt.partial(ori_prompt=ori_prompt)
    prompt_llm = agent_prompt | prompt_llm
    result = prompt_llm.invoke(state)
    logger.debug(f"##### Prompt agent result: [{result.content}]")
    keywords = json.loads(result.content)["keywords"]
    translation = json.loads(result.content)["translation"]
    logger.info(f"\n### Keywords: [{keywords}] \n### Translation: [{translation}]")
    return {
            "ori_prompt": ori_prompt, 
            "ind_prompt": translation, 
            "keywords": keywords,
            "mysql_list": [],
            "chroma_list": [],
            "combined_list": [],
            "result_list": [],
            }
