"""Define a custom Reasoning and Action agent with Human-in-the-Loop and Timeline support."""

from datetime import UTC, datetime
from typing import Dict, List, Literal, cast

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from react_agent.configuration import Configuration
from react_agent.state import InputState, State
from react_agent.tools import TOOLS
from react_agent.utils import load_chat_model, log_step
from react_agent.tools import tools_wrapper  


# Step 1: Call the model
async def call_model(state: State) -> Dict[str, List[AIMessage]]:
    log_step(state, "call_model")

    configuration = Configuration.from_context()
    model = load_chat_model(configuration.model).bind_tools(TOOLS)
    system_message = configuration.system_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )

    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state.messages]
        ),
    )

    if state.is_last_step and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                )
            ]
        }

    return {"messages": [response]}


# Step 2: Human-in-the-loop review before tool calls
async def human_review(state: State) -> dict:
    log_step(state, "human_review")

    last_msg = state.messages[-1]
    print("\n--- HUMAN REVIEW REQUIRED ---")
    print("AI wants to call tool(s):", getattr(last_msg, "tool_calls", None))
    approval = input("Approve tool call? (y/n): ").strip().lower()

    if approval != "y":
        from langchain_core.messages import AIMessage
        last_msg = AIMessage(
            content="Tool call rejected by human reviewer. Stopping here."
        )
        state.messages[-1] = last_msg
        return {"messages": [last_msg]}

    return {"messages": []}


# Step 3: Wrapper around tools to log timeline
async def tools_wrapper(state: State):
    log_step(state, "tools")
    return  ToolNode(TOOLS).invoke(state)


# Routing logic after model call
def route_after_model(state: State) -> Literal["__end__", "human_review"]:
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(f"Expected AIMessage, got {type(last_message).__name__}")
    if not last_message.tool_calls:
        return "__end__"
    return "human_review"


# Routing logic after human review
def route_after_human_review(state: State) -> Literal["__end__", "tools"]:
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(f"Expected AIMessage, got {type(last_message).__name__}")
    if not last_message.tool_calls:
        return "__end__"
    return "tools"


# === Build the LangGraph ===
builder = StateGraph(State, input=InputState, config_schema=Configuration)

builder.add_node("call_model", call_model)
builder.add_node("human_review", human_review)
builder.add_node("tools", tools_wrapper)

builder.add_edge("__start__", "call_model")

builder.add_conditional_edges("call_model", route_after_model)
builder.add_conditional_edges("human_review", route_after_human_review)
builder.add_edge("tools", "call_model")

graph = builder.compile(name="ReAct Agent with HITL and Timeline")
