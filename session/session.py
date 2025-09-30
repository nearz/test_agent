from langchain_core.messages import HumanMessage


class Session:
    def __init__(self, model: str, thread_id: str, graph):
        self.id = "temp"
        self.thread_id = thread_id
        self.model = model
        # NOTE: Maybe move graph to app level so it is compiled once on app start up?
        self.graph = graph

    def invoke(self, input: str):
        msg = HumanMessage(content=input)

        config = {"configurable": {"thread_id": self.thread_id}}
        context = {"llm": self.model}

        result = self.graph.invoke({"messages": [msg]}, config=config, context=context)
        return result["messages"][-1].content
