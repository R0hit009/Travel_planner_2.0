from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver

from osm_api import get_tourist_places_osm_by_name

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ------------------------------------------------------------------
# ENV / CLIENTS
# ------------------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# ------------------------------------------------------------------
# STATE
# ------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    summary: str


# ------------------------------------------------------------------
# TOOLS
# ------------------------------------------------------------------

@tool
def find_places(place: str, radius: int = 5000, top_n: int = 20):
    """
    Find tourist places using OpenStreetMap by place name.
    """
    return get_tourist_places_osm_by_name(place, radius, top_n)


tools = [find_places]

# ------------------------------------------------------------------
# MODELS
# ------------------------------------------------------------------

# Main agent model (tool-using)
agent_model = ChatOllama(
    model="llama3.1:8b"
).bind_tools(tools)

# Summariser model (separate on purpose)
summariser_model = ChatOllama(
    model="llama3.1:8b"
)

# ------------------------------------------------------------------
# AGENT NODE
# ------------------------------------------------------------------

def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content=(
            "You are a friendly travel assistant. "
            "Only call tools if they are necessary. "
            "For greetings or general advice, respond directly."
        )
    )

    response = agent_model.invoke(
        [system_prompt] + list(state["messages"])
    )

    return {"messages": [response]}

# ------------------------------------------------------------------
# ROUTING LOGIC
# ------------------------------------------------------------------

def should_continue(state: AgentState):
    last_message = state["messages"][-1]

    if not last_message.tool_calls:
        return "end"
    return "continue"

# ------------------------------------------------------------------
# SUMMARISER NODE (FINAL STEP)
# ------------------------------------------------------------------

def summariser_node(state: AgentState) -> AgentState:
    """
    Runs once at the end.
    Summarises the entire conversation.
    """

    summariser_prompt = SystemMessage(
        content=(
            "Summarise the conversation concisely. "
            "Focus on user intent, locations discussed, "
            "and any decisions or plans made."
        )
    )

    summary_response = summariser_model.invoke(
        [summariser_prompt] + list(state["messages"])
    )

    return {
        "summary": summary_response.content
    }

# ------------------------------------------------------------------
# GRAPH
# ------------------------------------------------------------------

graph = StateGraph(AgentState)

graph.add_node("our_agent", model_call)
graph.add_node("tools", ToolNode(tools=tools))
graph.add_node("summariser", summariser_node)

graph.add_edge(START, "our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": "summariser"
    }
)

graph.add_edge("tools", "our_agent")
graph.add_edge("summariser", END)

# ------------------------------------------------------------------
# MEMORY / APP
# ------------------------------------------------------------------

memory = InMemorySaver()

config = {
    "configurable": {
        "thread_id": "1"
    }
}

app = graph.compile(checkpointer=memory)

# ------------------------------------------------------------------
# OPTIONAL: GROQ SEARCH (UNCHANGED, STANDALONE)
# ------------------------------------------------------------------

def search(query: str, stream=True):
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": query}],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=True,
    )

    if stream:
        for chunk in completion:
            token = chunk.choices[0].delta.content or ""
            if token.strip():
                yield token
        return

    complete_answer = ""
    for chunk in completion:
        complete_answer += chunk.choices[0].delta.content or ""
    return complete_answer

# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("\n--- Travel Planner ---\n")

    user_input = input("Ask your question (or 'exit'): ")
    if user_input.lower() == "exit":
        exit()

    inputs = {
        "messages": [
            {"role": "user", "content": user_input}
        ]
    }

    events = app.stream(
        inputs,
        config,
        stream_mode="values"
    )

    final_state = None

    for event in events:
        final_state = event
        event["messages"][-1].pretty_print()

    print("\n--- SUMMARY (internal) ---")
    print(final_state.get("summary"))
