"""
Visualizes the existing LangGraph using Mermaid syntax.
"""

from main_ollama import app   # <-- change filename if needed


def visualize():
    graph_obj = app.get_graph()

    print("\n--- LANGGRAPH MERMAID DIAGRAM ---\n")
    print(graph_obj.draw_mermaid())


if __name__ == "__main__":
    visualize()
