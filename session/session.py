from langchain_core.messages import HumanMessage

from agent.state import build_graph


class Session:
    def __init__(self, model: str):
        self.id = "temp"
        self.model = model
        self.history = []
        self.graph = build_graph()

    def invoke(self, input: str):
        msg = HumanMessage(content=input)
        self.history.append(msg)
        result = self.graph.invoke(
            {"messages": self.history}, context={"llm": self.model}
        )
        self.history = result["messages"]
        return self.history[-1].content
