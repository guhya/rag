import logging
import mysql.connector

logger = logging.getLogger(__name__)

def get_mysql_conn():
    """Get MySQL DB Connection"""

    mysql_conn = mysql.connector.connect(
        host="localhost",
        user="sakila",
        password="sakila",
        database="sakila"
    )
    
    return mysql_conn
