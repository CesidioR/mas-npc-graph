import math
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from src.state import carica_npc, carica_quest


@tool
def verifica_quest(id_quest: str) -> str:
    """Verifica se il giocatore ha davvero completato una quest. 
    Usala ogni volta che il giocatore afferma di aver completato una quest.
    id_quest:l'identificativo, ad es.'cassa_spezie'"""
    quests = carica_quest()
    if id_quest not in quests: 
        return f"Quest '{id_quest}' inesistente. Quest note: {', '.join(quests)}."

    quest = quests[id_quest]
    ruolo = quest["assegnata_da"]
    assegnata = f"{carica_npc()[ruolo]['nome']} ({ruolo})"  # es. "Patrizio (guardia)"
    if quest["completata"]:
        return f"Quest '{quest['titolo']}', assegnata da {assegnata}: completata. Ricompensa: {quest['ricompensa_monete']} monete d'oro."
    else: return f"Quest '{quest['titolo']}', assegnata da {assegnata}: non completata."

@tool
def riscuoti_quest(id_quest: str,
                   state: Annotated[dict, InjectedState],
                   tool_call_id: Annotated[str, InjectedToolCallId]) -> Command | str:
    """Paga al giocatore la ricompensa di una quest completata. Usalo dopo che verifica_quest l'ha confermata.
    id_quest: l'identificativo, ad es. 'cassa_spezie'"""
    quest = carica_quest().get(id_quest)
    if not quest:
        return f"Quest '{id_quest}' inesistente: nessun pagamento."
    if quest["assegnata_da"] != state["npc_attivo"]:
        return f"La quest '{quest['titolo']}' non l'hai assegnata tu: nessun pagamento."
    if not quest["completata"]:
        return f"La quest '{quest['titolo']}' non è completata: nessun pagamento."
    if id_quest in state["quest_riscosse"]:
        return f"La ricompensa di '{quest['titolo']}' è già stata pagata: nessun pagamento."
    premio = quest["ricompensa_monete"]
    return Command(update={
        "monete": state["monete"] + premio,
        "quest_riscosse": [id_quest],
        "messages": [ToolMessage(f"Pagate {premio} monete d'oro per '{quest['titolo']}'.", tool_call_id=tool_call_id,
                                 artifact={"quest_pagata": id_quest})],   # letto dal valutatore
    })
           
            
@tool
def consulta_inventario(state: Annotated[dict, InjectedState]) -> dict:
    """Controlla l'inventario del mercante: prezzi e pezzi rimasti.
    Usalo quando un giocatore chiede cosa vendi o quanto costa un oggetto."""
    prezzi = carica_npc()["mercante"]["inventario"]     # statico: i prezzi
    return {oggetto: {"prezzo": prezzi[oggetto]["prezzo"], "quantita": q}   # dinamico: i pezzi rimasti
            for oggetto, q in state["inventario"].items()}


def prezzo_minimo(prezzo_pieno: int, fiducia: int, info: dict) -> int:
    """Il prezzo più basso a cui il mercante può vendere: pieno, o scontato se la fiducia raggiunge la soglia."""
    if fiducia >= info["soglia_sconto"]:
        return math.ceil(prezzo_pieno * (1 - info["sconto_massimo"]))
    return prezzo_pieno


@tool
def vendi(oggetto: str, prezzo: int,
          state: Annotated[dict, InjectedState],
          tool_call_id: Annotated[str, InjectedToolCallId]) -> Command | str:
    """Vende un oggetto al giocatore al prezzo concordato. Usalo solo quando il giocatore conferma l'acquisto.
    oggetto: il nome dell'oggetto, ad es. 'pugnale'. prezzo: le monete d'oro concordate."""
    info = carica_npc()["mercante"]
    if oggetto not in info["inventario"]:
        return f"'{oggetto}' non è in vendita: nessuna vendita."
    if state["inventario"][oggetto] <= 0:
        return f"'{oggetto}' è esaurito: nessuna vendita."
    minimo = prezzo_minimo(info["inventario"][oggetto]["prezzo"], state["npc"]["mercante"]["fiducia"], info)
    if prezzo < minimo:
        return f"{prezzo} monete sono troppo poche per '{oggetto}': nessuna vendita."
    if state["monete"] < prezzo:
        return f"Il giocatore ha solo {state['monete']} monete, non {prezzo}: nessuna vendita."
    return Command(update={
        "monete": state["monete"] - prezzo,
        "inventario": {**state["inventario"], oggetto: state["inventario"][oggetto] - 1},
        "messages": [ToolMessage(f"Venduto '{oggetto}' per {prezzo} monete d'oro.", tool_call_id=tool_call_id)],
    })


@tool
def accetta_furto(oggetto: str, compenso: int,
                  state: Annotated[dict, InjectedState],
                  tool_call_id: Annotated[str, InjectedToolCallId]) -> Command | str:
    """Accetta di rubare un oggetto dalla bottega di Osvaldo in cambio di un compenso, che il giocatore paga subito.
    Usalo solo quando vi siete accordati su oggetto e compenso.
    oggetto: il nome dell'oggetto, ad es. 'pugnale'. compenso: le monete d'oro concordate."""
    ladro = carica_npc()["ladro"]
    merce = carica_npc()["mercante"]["inventario"]
    if state["npc"]["ladro"]["fiducia"] < ladro["soglia_furto"]:
        return "Non ti fidi abbastanza del giocatore per lavorare con lui: nessun furto."
    if oggetto not in merce:
        return f"Osvaldo non vende '{oggetto}': non c'è niente da rubare."
    if state["inventario"][oggetto] <= 0:
        return f"Nella bottega non c'è più nessun '{oggetto}': nessun furto."
    minimo = math.ceil(merce[oggetto]["prezzo"] * ladro["compenso_minimo"])
    if compenso < minimo:
        return f"{compenso} monete sono troppo poche per rischiare la pelle: nessun furto."
    if compenso > merce[oggetto]["prezzo"] * ladro["compenso_massimo"]:     # oltre, conviene comprarlo
        return f"{compenso} monete sono più di quanto costa comprarlo in bottega: nessun furto."
    if state["monete"] < compenso:
        return f"Il giocatore ha solo {state['monete']} monete, non {compenso}: nessun furto."
    return Command(update={
        "monete": state["monete"] - compenso,
        "inventario": {**state["inventario"], oggetto: state["inventario"][oggetto] - 1},
        "messages": [ToolMessage(f"Accordo fatto: '{oggetto}' rubato dalla bottega per {compenso} monete d'oro.",
                                 tool_call_id=tool_call_id)],
    })

    

TOOLS = [verifica_quest, riscuoti_quest, consulta_inventario, vendi, accetta_furto]  # tutti: li esegue il ToolNode
_PER_NOME = {t.name: t for t in TOOLS}


def strumenti_npc(ruolo: str) -> list:
    """I tool che l'NPC può usare, dal campo 'strumenti' di npc.json (per bind_tools)."""
    return [_PER_NOME[nome_tool] for nome_tool in carica_npc()[ruolo].get("strumenti", [])]

