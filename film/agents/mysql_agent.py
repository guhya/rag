import logging

from film.agents.state import State
from utils import ew_mysql_util

logger = logging.getLogger(__name__)
mysql_conn = ew_mysql_util.get_mysql_conn()

def mysql_agent(state: State):
    ind_prompt = state["ind_prompt"]
    keywords = ind_prompt + " " + state["keywords"]
    
    cursor = mysql_conn.cursor(dictionary=True)
    qry = f"""
            SELECT
                film_id
                , MATCH(title, description) AGAINST('{keywords}') AS score
            FROM film_indonesia
            WHERE MATCH(title, description) AGAINST('{keywords}')
            LIMIT 5
            """
    logger.debug(f"### Natural language query : {qry}")

    cursor.execute(qry)    
    resultset = cursor.fetchall()
    
    # Normalize score
    max_mysql_score = max(rs["score"] for rs in resultset) if resultset else 0
    logger.debug(f"### Max mysql score: [{max_mysql_score}]")

    film_list = list()
    for rs in resultset:
        item_obj = {"item_id":rs["film_id"], "score":rs["score"] / max_mysql_score}
        film_list.append(item_obj)
    cursor.close()

    return {
            "mysql_list": film_list,
            }
