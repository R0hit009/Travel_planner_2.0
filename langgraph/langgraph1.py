from typing import TypedDict, List
from langgraph.graph import StateGraph
from IPython.display import display, Image

class AgentState(TypedDict):
    values : List[int]
    name : str
    result : str

def process_values(state: AgentState) -> AgentState:
    """This function handel multiple different types of inputs"""
    state['result'] = f"hi there {state["name"]}! your sum is {sum(state['values'])} "
    return state

graph = StateGraph(AgentState)
graph.add_node("processer" , process_values)
graph.set_entry_point("processer")
graph.set_finish_point("processer")

app = graph.compile()
display(Image(app.get_graph().draw_mermaid_png()))
