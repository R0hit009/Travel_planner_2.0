import json
from typing import TypedDict, Annotated, Sequence, Literal

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver

from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    AIMessage,
)
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from osm_api import get_tourist_places_osm_by_name


# =========================================================
# STATE
# =========================================================
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    intent: str


# =========================================================
# TOOL
# =========================================================
@tool
def find_places(place: str, radius: int = 5000, top_n: int = 20) -> str:
    """
    Find tourist places from OpenStreetMap by place name.
    """
    return get_tourist_places_osm_by_name(place, radius, top_n)


tools = [find_places]


# =========================================================
# MODELS
# =========================================================

# ❌ Wrong approach: one model for everything
# ✅ Correct approach: separate classifier + agent

intent_model = ChatOllama(
    model="llama3.1:8b",
    temperature=0,  # deterministic
)

agent_model = (
    ChatOllama(
        model="llama3.1:8b",
        temperature=0.2,
    )
    .bind_tools(tools)
)


# =========================================================
# INTENT CLASSIFIER NODE (LLM, NO TOOLS)
# =========================================================
INTENT_PROMPT = SystemMessage(
    content=(
        "Classify the user's intent into EXACTLY one of the following values:\n"
        "- greeting\n"
        "- travel_with_place\n"
        "- travel_no_place\n"
        "- planning\n"
        "- other\n\n"
        "Rules:\n"
        "- If a specific place name (city/country/location) is mentioned → travel_with_place\n"
        "- If travel is discussed but no place is mentioned → travel_no_place\n"
        "- Greetings like hi/hello → greeting\n"
        "- conversation like what to do tips → planning\n"
        "- Output ONLY valid JSON:\n"
        '{ "intent": "<value>" }'
    )
)

def intent_classifier(state: AgentState) -> AgentState:
    user_msg = state["messages"][-1]

    response = intent_model.invoke([INTENT_PROMPT, user_msg])

    try:
        intent = json.loads(response.content)["intent"]
    except Exception:
        intent = "other"

    return {"intent": intent}


# =========================================================
# INTENT ROUTER NODE (NO LLM)
# =========================================================
INTENT_PROMPTS = {
    "greeting": SystemMessage(
        content=(
            "User is greeting you.\n"
            "Respond briefly and politely.\n"
            "Ask what place they want to explore.\n"
            "Max 1 sentence."
        )
    ),
    "travel_no_place": SystemMessage(
        content=(
            "User wants travel suggestions but gave no place.\n"
            "Ask for the destination clearly and briefly.\n"
            "Max 1 sentence."
        )
    ),
    "planning": SystemMessage(
        content=(
            "User wants to plan what to do.\n"
            "if place provided plan accourding to that place\n"
            "otherwise ask for place or the type of mood.\n"
        )
    ),
    "other": SystemMessage(
        content=(
            "User input is unclear.\n"
            "Guide them to provide a travel destination.\n"
            "Max 1 sentence."
        )
    ),
}
responder_model = ChatOllama(
    model="llama3.1:8b",
    temperature=0.3,
)
def intent_router(state: AgentState) -> AgentState:
    intent = state["intent"]

    # travel_with_place → skip responder, go to agent
    if intent == "travel_with_place":
        return {}

    prompt = INTENT_PROMPTS.get(intent)

    if not prompt:
        return {}

    response = responder_model.invoke(
        [prompt, state["messages"][-1]]
    )

    return {"messages": [response]}



# =========================================================
# AGENT NODE (TOOLS ALLOWED)
# =========================================================
def agent_node(state: AgentState) -> AgentState:
    response = agent_model.invoke(state["messages"])
    return {"messages": [response]}


# =========================================================
# TOOL EXECUTION GUARD (CRITICAL)
# =========================================================
def should_continue(state: AgentState) -> Literal["continue", "end"]:
    last_message = state["messages"][-1]

    if not last_message.tool_calls:
        return "end"

    # hard validation — no guessing
    tool_call = last_message.tool_calls[0]
    args = tool_call.get("args", {})

    if not args.get("place"):
        return "end"

    return "continue"


# =========================================================
# GRAPH
# =========================================================
graph = StateGraph(AgentState)

graph.add_node("intent_classifier", intent_classifier)
graph.add_node("intent_router", intent_router)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "intent_classifier")
graph.add_edge("intent_classifier", "intent_router")

graph.add_conditional_edges(
    "intent_router",
    lambda state: state["intent"],
    {
        "greeting": END,
        "travel_no_place": END,
        "other": END,
        "travel_with_place": "agent",
    },
)

graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)

graph.add_edge("tools", "agent")


# =========================================================
# MEMORY + APP
# =========================================================
memory = InMemorySaver()
app = graph.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "travel-thread-1"}}


# =========================================================
# SYSTEM PROMPT (MAIN AGENT)
# =========================================================
system_message = SystemMessage(
    content=(
        "You are a concise travel assistant.\n"
        "- Respond in 1–2 sentences.\n"
        "- Call tools ONLY when a place name is explicitly mentioned.\n"
        "- Never guess locations.\n"
        "- Do not explain reasoning."
    )
)


# =========================================================
# REPL
# =========================================================
while True:
    print("\n----------------------------")
    user_input = input("Ask your question (or 'exit'): ")

    if user_input.lower() == "exit":
        break

    inputs = {
        "messages": [
            system_message,
            {"role": "user", "content": user_input},
        ]
    }

    for event in app.stream(inputs, config, stream_mode="values"):
        msg = event["messages"][-1]
        msg.pretty_print()
