import logging

from langchain_ollama import ChatOllama

from chatbot.state import State
from chatbot.tools import tool_registry
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

class Assistant:
    def __init__(self, llm: ChatOllama, prompt: ChatPromptTemplate):
        self.llm = llm
        self.prompt = prompt

    def __call__(self, state: State):        
        llm = self.llm
        prompt = self.prompt
        selected_tools = []
        if "selected_tools" in state and len(state["selected_tools"]) > 0:
            logger.debug(f"Selected Tools :\n{state["selected_tools"]}")
            for id in state["selected_tools"]:
                logger.debug(f"Adding tools : [{tool_registry[id]}]")
                selected_tools.append(tool_registry[id])                
        
        llm = prompt | llm.bind_tools(selected_tools)
        result = llm.invoke(state)
        if not result.tool_calls:
            return {"generation": result.content}
        else:
            return {"generation": "tool_calls", "messages": result}


