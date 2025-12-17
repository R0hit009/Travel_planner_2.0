from langchain_ollama import OllamaLLM
from typing import TypedDict, List , Union , Annotated , Sequence
from langgraph.graph import StateGraph , START, END
from langchain_core.messages import HumanMessage , AIMessage , BaseMessage , ToolMessage , SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
# Load Gemma 2 from Hugging Face

class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage],add_messages]

@tool
def add(a:list[int]):
    """this is a simple add function that takes a list of numbers and return the sum of them"""
    return sum(a)

@tool
def subtract(a:int , b:int):
    """this is a simple subtract function that subtract two numbers"""
    return a - b

@tool
def multiply(a:int , b:int):
    """this is a simple multiply function that multiply two numbers"""
    return a * b


tools = [add,subtract,multiply]


model = ChatOllama(model="llama3.1:8b").bind_tools(tools)

def model_call (state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content =  
        "You are my helpful AI assistant. "
        "Only call tools if the user explicitly asks you to calculate something. "
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


while True:
    print("\n\n----------------------------")
    user_input = input("Ask your question (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    inputs = {"messages": [{"role" : "user" , "content" : user_input}]}
    events = app.stream(inputs,
                        config,
                        stream_mode="values")
    for event in events:
         event["messages"][-1].pretty_print()
