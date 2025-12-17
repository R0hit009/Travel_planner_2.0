import requests
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from osm_api import get_tourist_places_osm_by_name  
# ======================
# 1. Define OSM Tool
# ======================

osm_tool = Tool(
    name="OpenStreetMapSearch",
    func=get_tourist_places_osm_by_name,
    description="Search for places and landmarks using OpenStreetMap. Input should be a place name."
)

# ======================
# 2. Define Model
# ======================
model = OllamaLLM(model="gemma2:9b")

# ======================
# 3. Agent with Tools
# ======================
agent = initialize_agent(
    tools=[osm_tool],
    llm=model,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    max_iterations=3,
    verbose=True
)

# ======================
# 4. Conversation Loop
# ======================
travel_data = """
- Manali: Peaceful mountain stay, trekking, snow activities.
- Jaipur: Monuments like Hawa Mahal, Amber Fort, City Palace.
- Kolkata: Victoria Memorial, Howrah Bridge, Dakshineswar Temple.
"""

template = """
You are a friendly travel assistant. 
Use the following travel information when helpful.
Travel Guide Data: {data}
User Question: {question}
If the user asks about a location, you may also use your tools.
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model | StrOutputParser()

while True:
    print("\n\n----------------------------")
    question = input("Ask your question (or 'exit' to quit): ")
    if question.lower() == "exit":
        break
    
    # Agent can decide to use OSM tool if needed
    result = chain.invoke(f"{question}.")
    print("\n🤖 Assistant:", result)


# from langchain_ollama import OllamaLLM
# from typing import TypedDict, List , Union , Annotated , Sequence
# from langgraph.graph import StateGraph , START, END
# from langchain_core.messages import HumanMessage , AIMessage , BaseMessage , ToolMessage , SystemMessage
# from langchain_core.tools import tool
# from langgraph.prebuilt import ToolNode
# from langgraph.graph.message import add_messages
# from langchain_ollama import ChatOllama
# from langgraph.checkpoint.memory import InMemorySaver
# from langgraph.prebuilt import create_react_agent
# from osm_api import get_tourist_places_osm_by_name
# # Load Gemma 2 from Hugging Face

# class AgentState(TypedDict):
#     messages : Annotated[Sequence[BaseMessage],add_messages]

# @tool
# def find_places(place:str , radius:int=5000 , top_n:int=20):
#     """
#     this tool find tourist places from open street map by place name 
#     and radius in meter and top_n number of places to return start 
#     the radius start by 5000 .
#     also sort the places by popularity or importance.
#     """
#     places_string = get_tourist_places_osm_by_name(place, radius,top_n)
#     return places_string

# tools = [find_places]


# model = ChatOllama(model="llama3.1:8b").bind_tools(tools)

# agent = create_react_agent(
#     model,
#     tools=tools,
#     prompt = SystemMessage(
#         content =  
#         "You are a friendly travel assistant."
#         "Only call tools if you deem it necessary to answer the user's query."
#         "For greetings, casual conversation, or general advice, respond directly without using tools."
#     ),
# ) 
# # def model_call (state: AgentState) -> AgentState:
# #     system_prompt = SystemMessage(
# #         content =  
# #         "You are a friendly travel assistant."
# #         "Only call tools if you deem it necessary to answer the user's query."
# #         "For greetings, casual conversation, or general advice, respond directly without using tools."
# #     )
# #     respose = model.invoke([system_prompt] + state['messages'])
# #     return {"messages": [respose]}

# def should_continue(state: AgentState) -> bool:
#     message = state['messages']
#     last_message = message[-1]
#     if not last_message.tool_calls:
#         return 'end'
#     else:
#         return 'continue'




# graph = StateGraph(AgentState)
# graph.add_node("our_agent" , model_call)

# too_node = ToolNode(tools=tools)
# graph.add_node("tools" , too_node)

# graph.add_edge(START , "our_agent")
# graph.add_conditional_edges(
#     "our_agent" , 
#     should_continue ,
#     {
#         'continue': "tools",
#         'end': END
#     }
# )

# graph.add_edge("tools" , "our_agent")

# memory = InMemorySaver()
# config = {"configurable": {"thread_id" : "1"}}
# app = graph.compile(checkpointer=memory)


# while True:
#     print("\n\n----------------------------")
#     user_input = input("Ask your question (or 'exit' to quit): ")
#     if user_input.lower() == 'exit':
#         break
#     inputs = {"messages": [{"role" : "user" , "content" : user_input}]}
#     events = app.stream(inputs,
#                         config,
#                         stream_mode="values")
#     print(events)
#     for event in events:
#          event["messages"][-1].pretty_print()
