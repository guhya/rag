import logging

logging.basicConfig(
    level = logging.DEBUG,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format = '%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s',  # Log message format
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler("log/chatbot.log")  # Log to a file
    ]
)