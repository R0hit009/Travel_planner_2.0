from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from typing import TypedDict, List , Union
from langgraph.graph import StateGraph , START, END
from langchain_core.messages import HumanMessage , AIMessage

# Load Gemma 2 from Hugging Face

class AgentState(TypedDict):
    messages : List[Union[HumanMessage,AIMessage]]

model = OllamaLLM(model="gemma2:9b")

def process(state: AgentState) -> AgentState:
    """This function create messeges and get respose from model"""
    respose = model.invoke(state['messages'])
    state['messages'].append(AIMessage(content=respose))
    print("\nAI: ", respose)
    return state

graph = StateGraph(AgentState)
graph.add_node("processer" , process)
graph.add_edge(START, "processer")
graph.add_edge("processer", END)   
app = graph.compile()

conversational_history = []
while True:
    print("\n\n----------------------------")
    user_input = input("Ask your question (or 'exit' to quit): ")
    conversational_history.append(HumanMessage(content=user_input))
    if user_input.lower() == 'exit':
        break
    result = app.invoke({"messages": conversational_history})
    conversational_history = result['messages']
