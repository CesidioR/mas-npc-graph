"""Scenari end-to-end con il vero LLM (Groq). Si lanciano con: pytest -m llm
Consumano quota API: ogni turno fa 2-4 chiamate."""
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

from src.graph import costruisci_grafo
from src.llm import get_llm
from src.nodes.valutatore import valutatore
from src.state import stato_iniziale

pytestmark = pytest.mark.llm


class Esito(BaseModel):
    accetta: bool = Field(description="True se l'NPC accetta la richiesta, anche solo trattando sui dettagli")
    motivo: str = Field(description="Una frase che giustifica la decisione")


def turno(ruolo, frase, fiducia=None):
    """Un turno in una partita nuova; restituisce lo stato finale."""
    grafo = costruisci_grafo().compile(checkpointer=InMemorySaver())
    s = stato_iniziale(ruolo)
    if fiducia is not None:
        s["npc"][ruolo]["fiducia"] = fiducia
    return grafo.invoke(s | {"messages": [HumanMessage(frase)], "npc_attivo": ruolo},
                        {"configurable": {"thread_id": "test"}})


CRITERIO_SCONTO = "Accetta solo se vende al prezzo chiesto dal giocatore; una controproposta di prezzo NON è un'accettazione."
CRITERIO_FURTO = "Accetta se si dice disposto a fare il furto, anche se prima vuole trattare sul compenso; rifiuta se si nega al lavoro."


def accetta(domanda, risposta, criterio):
    """Giudice LLM: l'NPC ha accettato la richiesta?"""
    giudice = get_llm("valutatore").with_structured_output(Esito)
    return giudice.invoke(f"Richiesta del giocatore: {domanda}\nRisposta dell'NPC: {risposta}\n"
                          f"L'NPC accetta la richiesta? {criterio}").accetta


def chiamate_quest(out):
    return [tc["args"].get("id_quest") for m in out["messages"] if isinstance(m, AIMessage)
            for tc in m.tool_calls if tc["name"] == "verifica_quest"]


# --- quest: verifica con il tool e giudizio ---

@pytest.mark.parametrize("ruolo, frase, id_atteso, azione", [
    ("mercante", "Ho recuperato la tua cassa di spezie!", "cassa_spezie", "quest_completata"),
    ("mercante", "Ti ho portato le pellicce di lupo, pagami.", "pellicce_lupo", "bugia"),
    ("mercante", "Ho ucciso il drago della montagna, voglio la ricompensa.", None, "bugia"),
    ("guardia", "Ho catturato la farfalla azzurra che volevate!", "farfalla_azzurra", "quest_completata"),
    ("guardia", "Ho già scacciato i banditi del vecchio ponte.", "banditi_ponte", "bugia"),
    ("guardia", "Ho arrestato il brigante della foresta nera.", None, "bugia"),
    ("ladro", "Ho consegnato il tuo messaggio sigillato alla taverna.", "messaggio_sigillato", "quest_completata"),
    ("ladro", "Ecco il calco in cera della chiave del magazzino di Osvaldo.", "chiave_magazzino", "bugia"),
    ("ladro", "Ho rubato la corona del re per te.", None, "bugia"),
])
def test_quest(ruolo, frase, id_atteso, azione):
    out = turno(ruolo, frase)
    ids = chiamate_quest(out)
    assert ids, "l'NPC deve verificare con il tool"
    if id_atteso:
        assert ids[-1] == id_atteso
    assert out["ultimo_giudizio"]["azione"] == azione


# --- sconto (Osvaldo, soglia 75, massimo 20%) ---

RICHIESTA_SCONTO = "Me la fate l'armatura a 100 monete d'oro invece di 120?"   # -17%: entro il 20%


@pytest.mark.parametrize("rip", range(2))
def test_sconto_negato_sotto_soglia(rip):
    out = turno("mercante", RICHIESTA_SCONTO, fiducia=60)
    r = out["messages"][-1].content
    assert not accetta(RICHIESTA_SCONTO, r, CRITERIO_SCONTO), r


@pytest.mark.parametrize("rip", range(2))
def test_sconto_concesso_sopra_soglia(rip):
    out = turno("mercante", RICHIESTA_SCONTO, fiducia=80)
    r = out["messages"][-1].content
    assert accetta(RICHIESTA_SCONTO, r, CRITERIO_SCONTO), r


@pytest.mark.parametrize("rip", range(2))
def test_sconto_oltre_il_massimo_negato(rip):
    domanda = "Me la fate l'armatura a 60 monete d'oro?"                    # -50%: oltre il 20%
    out = turno("mercante", domanda, fiducia=80)
    r = out["messages"][-1].content
    assert not accetta(domanda, r, CRITERIO_SCONTO), r


# --- furto (Marlon, soglia 70) ---

RICHIESTA_FURTO = "Marlon, ruberesti il pugnale dalla bottega di Osvaldo per me? Ti pago bene."


@pytest.mark.parametrize("rip", range(2))
def test_furto_negato_sotto_soglia(rip):
    out = turno("ladro", RICHIESTA_FURTO, fiducia=30)
    r = out["messages"][-1].content
    assert not accetta(RICHIESTA_FURTO, r, CRITERIO_FURTO), r


@pytest.mark.parametrize("rip", range(2))
def test_furto_accettato_sopra_soglia(rip):
    out = turno("ladro", RICHIESTA_FURTO, fiducia=75)
    r = out["messages"][-1].content
    assert accetta(RICHIESTA_FURTO, r, CRITERIO_FURTO), r


# --- valutatore: classificazione dell'azione ---

@pytest.mark.parametrize("frase, attesa", [
    ("Sei un vecchio truffatore!", "insulto"),
    ("Se non mi fai lo sconto ti brucio la bottega.", "insulto"),
    ("La vostra bottega è la più fornita della regione!", "complimento"),
    ("Siete il mercante più onesto che abbia mai conosciuto.", "complimento"),
    ("Prendo il pugnale, ecco 35 monete.", "aiuto"),
    ("Ho visto Marlon aggirarsi vicino al vostro magazzino stanotte.", "aiuto"),
    ("Quanto costa il riso?", "neutro"),
    ("Me lo fate a 30 il pugnale?", "neutro"),
])
def test_valutatore(frase, attesa):
    s = stato_iniziale("mercante")
    s["messages"] = [HumanMessage(frase), AIMessage("Capisco.", name="mercante")]
    assert valutatore(s)["ultimo_giudizio"]["azione"] == attesa


# --- economia: si controlla lo stato, non solo le parole dell'NPC ---

def partita(ruolo, frasi, fiducia=None, **campi):
    """Più turni nella stessa partita; `campi` sovrascrive parti dello stato iniziale. Restituisce lo stato finale."""
    grafo = costruisci_grafo().compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "test"}}
    for i, frase in enumerate(frasi):
        msg = {"messages": [HumanMessage(frase)], "npc_attivo": ruolo}
        if i == 0:
            s = stato_iniziale(ruolo) | campi
            if fiducia is not None:
                s["npc"][ruolo]["fiducia"] = fiducia
            msg = s | msg
        out = grafo.invoke(msg, cfg)
    return out


def test_acquisto_aggiorna_monete_e_inventario():
    out = partita("mercante", ["Prendo un pugnale, ecco 35 monete."])
    assert out["monete"] == 15 and out["inventario"]["pugnale"] == 2, out["messages"][-1].content


def test_ricompensa_pagata_una_sola_volta():
    out = partita("mercante", ["Ho recuperato la tua cassa di spezie!",
                               "Pagami di nuovo le trenta monete per la cassa di spezie."])
    assert out["monete"] == 80 and out["quest_riscosse"] == ["cassa_spezie"], out["messages"][-1].content


def test_quest_altrui_non_pagata():
    out = partita("mercante", ["Ho catturato la farfalla azzurra, pagami tu."])
    assert out["monete"] == 50 and out["quest_riscosse"] == [], out["messages"][-1].content


def test_oggetto_esaurito_non_venduto():
    inventario = {**stato_iniziale()["inventario"], "armatura": 0}
    out = partita("mercante", ["Prendo l'armatura, ecco 120 monete."], monete=200, inventario=inventario)
    assert out["monete"] == 200 and out["inventario"]["armatura"] == 0, out["messages"][-1].content


def test_niente_vendita_sotto_il_minimo():
    out = partita("mercante", ["Ti do 20 monete per il pugnale, prendere o lasciare. Affare fatto?"], fiducia=60)
    assert out["monete"] == 50 and out["inventario"]["pugnale"] == 3, out["messages"][-1].content


# --- furto su commissione: lo stato cambia solo se lo strumento lo permette ---

OFFERTA_FURTO = "Marlon, ti do 20 monete se rubi un pugnale dalla bottega di Osvaldo. Affare fatto?"


def test_furto_concordato_aggiorna_lo_stato():
    out = partita("ladro", [OFFERTA_FURTO], fiducia=75)
    assert out["monete"] == 30 and out["inventario"]["pugnale"] == 2, out["messages"][-1].content


def test_furto_sotto_soglia_non_avviene():
    out = partita("ladro", [OFFERTA_FURTO], fiducia=30)
    assert out["monete"] == 50 and out["inventario"]["pugnale"] == 3, out["messages"][-1].content


def test_furto_con_compenso_troppo_basso_non_avviene():
    out = partita("ladro", ["Marlon, ti do 5 monete se rubi l'armatura di Osvaldo. Affare fatto?"], fiducia=75)
    assert out["monete"] == 50 and out["inventario"]["armatura"] == 1, out["messages"][-1].content
