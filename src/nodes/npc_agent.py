from langchain_core.messages import SystemMessage

from src.llm import get_llm
from src.prompts import system_prompt
from src.state import WorldState
from src.tools import strumenti_npc
from langchain_core.messages import AIMessage, HumanMessage

def messaggi_visibili(messages: list, ruolo: str) -> list:
    """I messaggi che l'NPC `ruolo` può vedere: i suoi turni passati più il turno corrente."""
    # 1. dividi in turni: un turno nuovo a ogni messaggio del giocatore
    turni = []
    for m in messages:
        if isinstance(m, HumanMessage) or not turni:
            turni.append([m])
        else:
            turni[-1].append(m)
    if not turni:
        return []

    # 2. tieni i turni passati in cui ha risposto `ruolo` (grazie alla firma name)
    passati = [t for t in turni[:-1]
               if any(isinstance(m, AIMessage) and m.name == ruolo for m in t)]

    # 3. turni tenuti + turno corrente, in un'unica lista piatta
    return [m for turno in passati + [turni[-1]] for m in turno]



def npc_agent(state: WorldState) -> dict:
    """Fa parlare l'NPC attivo: risponde al giocatore o chiama un tool."""
    ruolo = state["npc_attivo"]
    llm = get_llm("dialogo").bind_tools(strumenti_npc(ruolo))
    risposta = llm.invoke([SystemMessage(system_prompt(ruolo, state))] + messaggi_visibili(state["messages"], ruolo))
    risposta.name = ruolo  # firma: chi ha detto questa battuta (riepilogo, e in Fase 5 filtri per NPC)
    return {"messages": [risposta]}
