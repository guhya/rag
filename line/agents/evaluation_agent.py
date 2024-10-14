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
    
    user_prompt = state["ori_prompt"]
    evaluation_llm = ChatOllama(model="llama3.1:8b-instruct-q8_0", temperature = 0, format="json")
    agent_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are evaluator agent tasked to evaluate if a review from the user is positive or negative.
                If review is negative, you will need to analyze it and check if the review is related to technical problem or not. 
                Technical problem is caused by user frustration in using the app, can be app error or underlying system problem. 
                A frustation because of poor service is not technical problem.
                When user mention cashback, or debit cards, it means he is talking about banking service and product, and not about the app.
                
                Example of technical problem is:
                 - Slow connectivity, error, frustration with the way the app works, or bad UI/UX.
                 - Unresponsive app
                 - Confusing app navigation
                 
                Thi is not technical problem:
                 - Poor banking service
                 - Slow service, such as delay in sending debit card
                 - Feeling cheated or frustration because of misleading information
                 - Debit card problem
                 - Cashback problem
                 
                You will be given a rating from the user and the review content in Indonesian language for you to assess.
                
                Score:
                A rating above 3 indicate positive experience.
                A score of positive means that user is satisfied, negative if the user is frustrated or angry or overal had negative experience.
                Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct.
                """
            ),
            (            
                "human", 
                """
                Given the prompt: {user_prompt} give your assesment and classify the problem if any.
                Return JSON with three keys, 
                binary_score_experience is 'positive' or 'negative' score to indicate the experience user had. 
                binary_score_is_technical is 'true' or 'false' score to indicate if the negative experience is caused by technical problem. 
                And a key, explanation, that contains an explanation of your assesment.                
                """
            )
        ]
    )    
    agent_prompt = agent_prompt.partial(user_prompt=user_prompt)
    evaluation_llm = agent_prompt | evaluation_llm
    result = evaluation_llm.invoke(state)
    binary_score_experience = json.loads(result.content)["binary_score_experience"]
    binary_score_is_technical = json.loads(result.content)["binary_score_is_technical"]
    explanation = json.loads(result.content)["explanation"]
    logger.info(f"Prompt: {user_prompt}\n AI: {result.content}")

    return {
            "evaluation": [binary_score_experience, binary_score_is_technical, explanation]
            }
