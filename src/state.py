import json
from operator import add
from pathlib import Path
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import MessagesState

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MONETE_INIZIALI = 50
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
    monete: int
    inventario: dict[str, int]
    quest_riscosse: Annotated[list[str], add]


def carica_npc() -> dict:
    """Dati statici degli NPC (personalità, stile, relazioni, inventario)."""
    with open(DATA_DIR / "npc.json", encoding="utf-8") as f:
        return json.load(f)["npc"]


def carica_quest() -> dict:
    """Quest del mondo (id → titolo, descrizione, assegnata_da, ricompensa, completata)."""
    with open(DATA_DIR / "quests.json", encoding="utf-8") as f:
        return json.load(f)["quests"]


def delta_personale(delta: float, info: dict) -> int:
    """Δ di fiducia filtrato dal carattere dell'NPC: moltiplicato per la sua sensibilità ai Δ positivi o negativi."""
    s = info.get("sensibilita", {})
    return int(delta * s.get("positivo" if delta > 0 else "negativo", 1.0))


# Con le sole parole la fiducia sale fino a qui, sotto le soglie di sconto (75) e furto (70):
# per superarlo servono le quest completate, le uniche azioni verificate da un tool.
TETTO_PAROLE = 65
AZIONI_A_PAROLE = {"complimento", "aiuto"}


def aggiorna_fiducia(fiducia: int, delta: int, azione: str) -> int:
    """Nuova fiducia: clamp 0-100 e, per complimenti e aiuti, al massimo TETTO_PAROLE."""
    nuova = max(0, min(100, fiducia + delta))
    if azione in AZIONI_A_PAROLE and delta > 0:
        nuova = min(nuova, max(TETTO_PAROLE, fiducia))  # non abbassa chi è già sopra il tetto
    return nuova


def stato_iniziale(npc_attivo: str = "mercante") -> WorldState:
    """World State di una partita nuova, costruito da data/npc.json."""
    dati = carica_npc()
    inventario = dati["mercante"]["inventario"]
    return WorldState(
        messages=[],
        npc_attivo=npc_attivo,
        npc={
            nome: NPCState(fiducia=info["fiducia_iniziale"], umore=info["umore_iniziale"])
            for nome, info in dati.items()
        },
        eventi=[],
        ultimo_giudizio=None,
        monete=MONETE_INIZIALI,
        inventario={oggetto: info["quantita"] for oggetto, info in inventario.items()},
        quest_riscosse=[]
    )
