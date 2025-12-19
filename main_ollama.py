from langchain_ollama import OllamaLLM
from typing import TypedDict, List , Union , Annotated , Sequence
from langgraph.graph import StateGraph , START, END
from langchain_core.messages import HumanMessage , AIMessage , BaseMessage , ToolMessage , SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from osm_api import get_tourist_places_osm_by_name
from typing import Iterator, Dict, Any
from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()
# Load Gemma 2 from Hugging Face

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage],add_messages]

@tool
def find_places(place:str , radius:int=5000 , top_n:int=20):
    """
    this tool find tourist places from open street map by place name 
    and radius in meter and top_n number of places to return start 
    the radius start by 5000 .
    also sort the places by popularity or importance.
    """
    places_string = get_tourist_places_osm_by_name(place, radius,top_n)
    return places_string

tools = [find_places]


model = ChatOllama(model="llama3.1:8b").bind_tools(tools)

def model_call (state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content =  
        "You are a friendly travel assistant."
        "Only call tools if you deem it necessary to answer the user's query."
        "For greetings, casual conversation, or general advice, respond directly without using tools."
    )
    respose = model.invoke([system_prompt] + state['messages'])
    return {"messages": [respose]}

def should_continue(state: AgentState) -> bool:
    message = state['messages']
    last_message = message[-1]
    if not last_message.tool_calls:
        return 'end'
    else:
        return 'continue'




graph = StateGraph(AgentState)
graph.add_node("our_agent" , model_call)

too_node = ToolNode(tools=tools)
graph.add_node("tools" , too_node)

graph.add_edge(START , "our_agent")
graph.add_conditional_edges(
    "our_agent" , 
    should_continue ,
    {
        'continue': "tools",
        'end': END
    }
)

graph.add_edge("tools" , "our_agent")

memory = InMemorySaver()
config = {"configurable": {"thread_id" : "1"}}
app = graph.compile(checkpointer=memory)

def search(query:str,stream=True):
    completion = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "user",
        "content": query}
    ],
    temperature=1,
    max_completion_tokens=1024,
    top_p=1,
    # reasoning_effort="medium",
    stream=True,
    stop=None
)
    
    if stream:
        for chunk in completion:
            token = chunk.choices[0].delta.content or ""
            if token.strip():
                yield token
        return

    else:
        complete_answer=""
    
        for chunk in completion:
            complete_answer += chunk.choices[0].delta.content or "" + "\n"

        return complete_answer
    

if __name__ == "__main__":
    print("\n\n----------------------------")
    user_input = input("Ask your question (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        exit()
    inputs = {"messages": [{"role" : "user" , "content" : user_input}]}
    events = app.stream(inputs,
                        config,
                        stream_mode="values")
    print(events)
    for event in events:
         print(event)
         event["messages"][-1].pretty_print()
