from langchain_openai import ChatOpenAI
from langgraph.graph.state import Runnable

from .tools import get_tools


# NOTE: How to make generic for various models.
def get_model_with_tools(model: str) -> Runnable:
    return ChatOpenAI(model=model).bind_tools(get_tools())
