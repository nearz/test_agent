from typing import Annotated, TypedDict, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, START
from langgraph.graph.state import Runnable
from langgraph.prebuilt import ToolNode

from .tools import get_tools


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    llm: Runnable


def call_llm(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        "You are an AI assistant, please answer my query to the best of your ability."
    )
    all_msgs = [system_prompt] + list(state["messages"])
    response = state["llm"].invoke(all_msgs)
    return {"messages": list(state["messages"]) + [response], "llm": state["llm"]}


def should_continue(state: AgentState) -> bool:
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage):
        if not last_msg.tool_calls:
            return False
        else:
            return True


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("call_llm", call_llm)
    graph.add_node("tools", ToolNode(tools=get_tools()))

    graph.add_edge(START, "call_llm")
    graph.add_conditional_edges(
        "call_llm",
        should_continue,
        {
            True: "tools",
            False: END,
        },
    )
    graph.add_edge("tools", "call_llm")

    return graph.compile()
