"""Test del grafo completo con un LLM finto: nessuna chiamata a Groq, nessuna quota consumata.

Le risposte del modello sono scritte in anticipo, quindi si verifica il cablaggio
(ciclo ReAct, valutatore, gossip, stato, persistenza) e non la qualità delle risposte.
"""
import sqlite3

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

import src.nodes.npc_agent as npc_agent
import src.nodes.valutatore as valutatore
from src.graph import costruisci_grafo
from src.nodes.valutatore import Valutazione
from src.state import stato_iniziale


class LLMFinto:
    """Sostituto di get_llm(): restituisce in ordine le risposte preparate e ricorda cosa ha ricevuto."""

    def __init__(self, risposte):
        self.risposte = list(risposte)
        self.ricevuti = []

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        self.ricevuti.append(messages)
        return self.risposte.pop(0)


def chiamata(nome, args, id_):
    return AIMessage("", tool_calls=[{"name": nome, "args": args, "id": id_}])


@pytest.fixture
def finti(monkeypatch):
    """Installa due LLM finti, uno per gli NPC e uno per il valutatore; il test li riempie di risposte."""
    dialogo, giudice = LLMFinto([]), LLMFinto([])
    monkeypatch.setattr(npc_agent, "get_llm", lambda ruolo: dialogo)
    monkeypatch.setattr(valutatore, "get_llm", lambda ruolo: giudice)
    return dialogo, giudice


def gioca(grafo, config, ruolo, frase, nuovo=False):
    msg = {"messages": [HumanMessage(frase)], "npc_attivo": ruolo}
    if nuovo:
        msg = stato_iniziale(ruolo) | msg
    return grafo.invoke(msg, config)


QUEST_CASSA = [
    chiamata("verifica_quest", {"id_quest": "cassa_spezie"}, "c1"),
    chiamata("riscuoti_quest", {"id_quest": "cassa_spezie"}, "c2"),
    AIMessage("Ecco le vostre 30 monete, amico mio!"),
]
GIUDIZIO_CASSA = Valutazione(azione="quest_completata", umore="gioioso",
                             evento="Il giocatore ha recuperato la cassa di spezie per Osvaldo.")


def test_turno_completo_con_ciclo_react(finti):
    dialogo, giudice = finti
    dialogo.risposte, giudice.risposte = list(QUEST_CASSA), [GIUDIZIO_CASSA]
    grafo = costruisci_grafo().compile(checkpointer=InMemorySaver())
    out = gioca(grafo, {"configurable": {"thread_id": "t"}}, "mercante", "Ho la cassa di spezie!", nuovo=True)

    # ciclo ReAct: due giri di tool, poi la risposta finale firmata
    tipi = [type(m).__name__ for m in out["messages"]]
    assert tipi == ["HumanMessage", "AIMessage", "ToolMessage", "AIMessage", "ToolMessage", "AIMessage"]
    assert out["messages"][-1].name == "mercante"
    assert "completata" in out["messages"][2].content
    # riscuoti_quest ha modificato lo stato
    assert out["monete"] == 80 and out["quest_riscosse"] == ["cassa_spezie"]
    # valutatore: +15 a Osvaldo; gossip: alleato +11 (7.5 x 1.5), rivale -9 (7.5 x 1.2)
    assert {r: s["fiducia"] for r, s in out["npc"].items()} == {"mercante": 75, "guardia": 61, "ladro": 21}
    assert out["npc"]["mercante"]["umore"] == "gioioso"
    assert sorted((e["npc"], e["fonte"]) for e in out["eventi"]) == [
        ("guardia", "sentito dire"), ("ladro", "sentito dire"), ("mercante", "vissuto")]


def test_un_npc_non_vede_gli_scambi_degli_altri(finti):
    dialogo, giudice = finti
    dialogo.risposte = list(QUEST_CASSA) + [AIMessage("Non ne so nulla, viandante.")]
    giudice.risposte = [GIUDIZIO_CASSA, Valutazione(azione="neutro", umore="sereno", evento=None)]
    grafo = costruisci_grafo().compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t"}}
    gioca(grafo, config, "mercante", "Ho la cassa di spezie!", nuovo=True)
    out = gioca(grafo, config, "guardia", "Osvaldo mi ha pagato?")

    visti = dialogo.ricevuti[-1][1:]             # cosa ha ricevuto il modello per Patrizio (senza il prompt)
    assert [m.content for m in visti] == ["Osvaldo mi ha pagato?"]
    assert "Il giocatore ha recuperato la cassa di spezie" in dialogo.ricevuti[-1][0].content   # la voce sì
    assert len(out["messages"]) == 8             # lo stato conserva comunque tutta la conversazione
    assert out["npc"]["guardia"]["fiducia"] == 61   # il turno neutro non ripropaga il giudizio precedente


def test_ripresa_della_partita_dopo_il_riavvio(finti, tmp_path):
    """Memoria persistente: la partita sopravvive alla chiusura del programma (stesso thread_id)."""
    dialogo, giudice = finti
    dialogo.risposte = list(QUEST_CASSA) + [AIMessage("Vi ho già dato trenta monete.")]
    giudice.risposte = [GIUDIZIO_CASSA, Valutazione(azione="neutro", umore="gioioso", evento=None)]
    db = tmp_path / "partite.db"
    config = {"configurable": {"thread_id": "partita-di-prova"}}

    # prima sessione: si gioca un turno e si chiude la connessione, come uscendo dal programma
    conn = sqlite3.connect(db, check_same_thread=False)
    gioca(costruisci_grafo().compile(checkpointer=SqliteSaver(conn)), config,
          "mercante", "Ho la cassa di spezie!", nuovo=True)
    conn.close()

    # seconda sessione: connessione e grafo nuovi, nessuno stato in memoria
    conn = sqlite3.connect(db, check_same_thread=False)
    grafo = costruisci_grafo().compile(checkpointer=SqliteSaver(conn))
    ripreso = grafo.get_state(config).values
    assert len(ripreso["messages"]) == 6
    assert ripreso["monete"] == 80 and ripreso["quest_riscosse"] == ["cassa_spezie"]
    assert ripreso["npc"]["mercante"]["fiducia"] == 75

    # e si continua a giocare: l'NPC riceve la conversazione della sessione precedente
    out = gioca(grafo, config, "mercante", "Quante monete mi hai dato?")
    assert [m.content for m in dialogo.ricevuti[-1][1:]][0] == "Ho la cassa di spezie!"
    assert len(out["messages"]) == 8 and out["monete"] == 80
    conn.close()

    # un thread_id diverso è una partita diversa
    conn = sqlite3.connect(db, check_same_thread=False)
    altra = costruisci_grafo().compile(checkpointer=SqliteSaver(conn)).get_state({"configurable": {"thread_id": "altra"}})
    assert altra.values == {}
    conn.close()


def test_la_stessa_quest_vale_fiducia_una_volta_sola(finti):
    """Dichiarare di nuovo una quest già pagata non deve dare altro +15."""
    dialogo, giudice = finti
    dialogo.risposte = list(QUEST_CASSA) + [
        chiamata("verifica_quest", {"id_quest": "cassa_spezie"}, "c3"),
        chiamata("riscuoti_quest", {"id_quest": "cassa_spezie"}, "c4"),     # rifiutato: già pagata
        AIMessage("Vi ho già pagato, amico mio."),
    ]
    giudice.risposte = [GIUDIZIO_CASSA, GIUDIZIO_CASSA]                     # il giudice ci casca due volte
    grafo = costruisci_grafo().compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t"}}
    gioca(grafo, config, "mercante", "Ho la cassa di spezie!", nuovo=True)
    out = gioca(grafo, config, "mercante", "Ho la cassa di spezie, te lo ricordo!")
    assert out["npc"]["mercante"]["fiducia"] == 75 and out["monete"] == 80
    assert out["ultimo_giudizio"]["azione"] == "neutro"
    assert len(out["eventi"]) == 3                                          # nessun evento ripetuto


def test_quest_confermata_ma_non_pagata_non_da_fiducia(finti):
    dialogo, giudice = finti
    dialogo.risposte = [chiamata("verifica_quest", {"id_quest": "cassa_spezie"}, "c1"),
                        AIMessage("Ottimo lavoro, viaggiatore!")]              # l'NPC si dimentica di pagare
    giudice.risposte = [GIUDIZIO_CASSA]
    grafo = costruisci_grafo().compile(checkpointer=InMemorySaver())
    out = gioca(grafo, {"configurable": {"thread_id": "t"}}, "mercante", "Ho la cassa di spezie!", nuovo=True)
    assert out["npc"]["mercante"]["fiducia"] == 60 and out["ultimo_giudizio"]["azione"] == "neutro"
