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
    prompt_llm = ChatOllama(model="llama3.1:8b-instruct-q8_0", temperature = 0, format="json")
    agent_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                I am running a movie database service in Indonesian language where user can search movie based on prompt they entered.
                The prompt is not descriptive enough and we need to transform it into comma separated keywords.
                You will be a prompt assistant whose job is to extract keywords from the user prompt. 
                The keywords is comma separated text, which must be related to the prompt.                

                Reminder:
                - Make sure to only include the most relevant keywords.
                - Make sure that the keywords are comma separated.
                - Make sure all keywords are in Indonesian language.            
                - Return in plain text like for example: cantik, penyanyi, musik, pertunjukan
                - Do not write an introduction or summary in your response. 
                
                Current time: {time}.
                """
            ),
            (            
                "human", 
                """
                Given the prompt: {ori_prompt}, extract comma separated keywords related to this prompt.                
                Return JSON with two two keys, 
                keywords: keywords extracted. 
                And a key, translation, user prompt translated into Indonesian language.                
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
