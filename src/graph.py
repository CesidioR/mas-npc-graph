from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.nodes.npc_agent import npc_agent
from src.nodes.valutatore import valutatore
from src.state import WorldState
from src.tools import TOOLS


def costruisci_grafo() -> StateGraph:
    """Struttura del grafo, senza checkpointer: lo aggiunge chi lo compila."""
    builder = StateGraph(WorldState)
    builder.add_node("npc_agent", npc_agent)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_node("valutatore", valutatore)

    builder.add_edge(START, "npc_agent")
    builder.add_conditional_edges("npc_agent", tools_condition, {"tools": "tools", END: "valutatore"})
    builder.add_edge("tools", "npc_agent") 
    builder.add_edge("valutatore", END)
    return builder


graph = costruisci_grafo().compile() 