from dataclasses import dataclass
from typing import Annotated, TypedDict, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from .tools import get_tools
from .model import get_model_with_tools


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@dataclass
class ContextSchema:
    llm: str


def call_llm(state: AgentState, runtime: Runtime[ContextSchema]) -> AgentState:
    system_prompt = SystemMessage(
        "You are an AI assistant, please answer my query to the best of your ability."
    )
    llm = get_model_with_tools(runtime.context.llm)
    all_msgs = [system_prompt] + list(state["messages"])
    response = llm.invoke(all_msgs)
    return {"messages": [response]}


def should_continue(state: AgentState) -> bool:
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage):
        if not last_msg.tool_calls:
            return False
        else:
            return True


def build_graph():
    graph = StateGraph(AgentState, context_schema=ContextSchema)
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
