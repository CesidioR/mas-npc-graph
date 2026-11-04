from langchain_core.messages import SystemMessage

from src.llm import get_llm
from src.prompts import system_prompt
from src.state import WorldState
from src.tools import strumenti_npc


def npc_agent(state: WorldState) -> dict:
    """Fa parlare l'NPC attivo: risponde al giocatore o chiama un tool."""
    nome = state["npc_attivo"]
    llm = get_llm("dialogo").bind_tools(strumenti_npc(nome))
    risposta = llm.invoke([SystemMessage(system_prompt(nome, state))] + state["messages"])
    risposta.name = nome  # firma: chi ha detto questa battuta (riepilogo, e in Fase 5 filtri per NPC)
    return {"messages": [risposta]}
