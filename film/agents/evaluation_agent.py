import logging
import json
from datetime import date, datetime

from film.agents.state import State

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from utils import ew_mysql_util
mysql_conn = ew_mysql_util.get_mysql_conn()

logger = logging.getLogger(__name__)

    
def evaluation_agent(state: State):
    
    ind_prompt = state["ind_prompt"]
    user_prompt = ind_prompt + " " + state["keywords"]
    logger.debug(f"### User prompt: [{user_prompt}]")
    logger.debug(f"### State :\n{state}")
    
    items = state["combined_list"]
    result_list = []
    for item in items[:5]:
        logger.debug(f"### Get item description: [{item["item_id"]}]")
        # llama3.2:3b-instruct-q8_0
        # llama3.1:8b-instruct-q8_0
        evaluation_llm = ChatOllama(model="llama3.2:3b-instruct-q8_0", temperature = 0, format="json")
        agent_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    I am running a movie database service in Indonesian language where user can search movie based on prompt they entered.
                    You will be a evaluator assistant whose job is to asses if the description matches the theme mentioned in the prompt.
                    
                    Score:
                    A score of yes means that the item has the theme specified in the prompt, even if it is minor. 
                    Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct.
                    Current time: {time}.
                    """
                ),
                (            
                    "human", 
                    """
                    Given the prompt: {user_prompt} and the item description: {description} assess if the description has the theme in the prompt.
                    Return JSON with two two keys, 
                    binary_score is 'yes' or 'no' score to indicate that the answer address the question. 
                    And a key, explanation, that contains an explanation of the score.                
                    """
                )
            ]
        )    
        item_description = get_description(item["item_id"])    
        agent_prompt = agent_prompt.partial(time=datetime.now())
        agent_prompt = agent_prompt.partial(user_prompt=user_prompt)
        agent_prompt = agent_prompt.partial(description=item_description)
        evaluation_llm = agent_prompt | evaluation_llm
        result = evaluation_llm.invoke(state)
        logger.debug(f"##### Evaluation agent result: \n[{result.content}] \nDescription: {item_description}")
        ai_response = json.loads(result.content)["binary_score"]
        if "yes" == ai_response:
            item["description"] = item_description
            result_list.append(item)
    
    return {
            "result_list": result_list,
            }


def get_description(item_id):
    """Add additional context to a keywords to enhance result"""

    # Get additional data from MySQL
    cursor = mysql_conn.cursor(dictionary=True)
    qry = f"SELECT film_id, title, description FROM film_indonesia WHERE film_id IN ({item_id})"
    cursor.execute(qry)    
    resultset = cursor.fetchall()
    description = ""
    for rs in resultset:
        description = rs["title"] + ". " + rs["description"]
    cursor.close()

    return description
