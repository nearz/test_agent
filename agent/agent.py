from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from .state import build_graph
from .tools import get_tools


class Agent:
    def __init__(self, model: str):
        self.model = ChatOpenAI(model=model).bind_tools(
            get_tools()
        )  # Note: how ot make generic
        self.history = []
        self.graph = build_graph()

    def invoke(self, input: str):
        msg = HumanMessage(content=input)
        self.history.append(msg)
        result = self.graph.invoke({"llm": self.model, "messages": self.history})
        self.history = result["messages"]
        return self.history[-1].content
