import json
from operator import add
from pathlib import Path
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import MessagesState

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
Umore = Literal["arrabbiato", "sereno", "sospettoso", "gioioso"]

class NPCState(TypedDict):
    """Stato degli NPC"""
    fiducia: int  # 0-100
    umore: Umore
    
class Giudizio(TypedDict):
    ruolo: str 
    azione: str
    delta: int
    evento: str | None
    

class Evento(TypedDict):
    npc: str
    testo: str
    fonte: Literal["vissuto", "sentito dire"]


class WorldState(MessagesState):
    """Stato del mondo di gioco"""
    npc_attivo: str
    npc: dict[str, NPCState]
    eventi: Annotated[list[Evento], add]  # memoria a lungo termine 
    ultimo_giudizio: Giudizio | None


def carica_npc() -> dict:
    """Dati statici degli NPC (personalità, stile, relazioni, inventario)."""
    with open(DATA_DIR / "npc.json", encoding="utf-8") as f:
        return json.load(f)["npc"]


def carica_quest() -> dict:
    """Quest del mondo (id → titolo, descrizione, assegnata_da, ricompensa, completata)."""
    with open(DATA_DIR / "quests.json", encoding="utf-8") as f:
        return json.load(f)["quests"]


def stato_iniziale(npc_attivo: str = "mercante") -> WorldState:
    """World State di una partita nuova, costruito da data/npc.json."""
    dati = carica_npc()
    return WorldState(
        messages=[],
        npc_attivo=npc_attivo,
        npc={
            nome: NPCState(fiducia=info["fiducia_iniziale"], umore=info["umore_iniziale"])
            for nome, info in dati.items()
        },
        eventi=[],
        ultimo_giudizio=None
    )
