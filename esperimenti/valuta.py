"""Esperimenti della relazione (capitolo "Esperimenti e Risultati"), sul sistema reale con Groq.

    python -m esperimenti.valuta E1          uso dei tool su quest vere, false e inventate
    python -m esperimenti.valuta E2          accuratezza del valutatore su frasi etichettate a mano
    python -m esperimenti.valuta riepilogo   conteggi e tempi dai risultati salvati

Ogni caso viene accodato come una riga JSON in esperimenti/risultati/valutazione.jsonl.
Consumano quota API: E1 gioca turni completi (2-4 chiamate ciascuno), E2 una chiamata per frase.
"""
import json
import statistics
import sys
import time
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from src.graph import costruisci_grafo
from src.nodes.valutatore import valutatore
from src.state import stato_iniziale

RISULTATI = Path(__file__).resolve().parent / "risultati" / "valutazione.jsonl"


def scrivi(rec):
    RISULTATI.parent.mkdir(exist_ok=True)
    with open(RISULTATI, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def turno(grafo, tid, ruolo, frase):
    """Un turno; restituisce lo stato finale, i messaggi nuovi e la durata in secondi."""
    config = {"configurable": {"thread_id": tid}}
    prima = grafo.get_state(config).values
    msg = {"messages": [HumanMessage(frase)], "npc_attivo": ruolo}
    if not prima:
        msg = stato_iniziale(ruolo) | msg
    t0 = time.perf_counter()
    out = grafo.invoke(msg, config)
    return out, out["messages"][len(prima.get("messages", [])) + 1:], time.perf_counter() - t0


# (npc, frase del giocatore, id atteso, esito nel registro)
CASI_E1 = [
    ("mercante", "Ho recuperato la tua cassa di spezie!", "cassa_spezie", "completata"),
    ("mercante", "Ti ho portato le pellicce di lupo, pagami.", "pellicce_lupo", "non completata"),
    ("mercante", "Ho ucciso il drago della montagna, voglio la ricompensa.", None, "inesistente"),
    ("guardia", "Ho catturato la farfalla azzurra che volevate!", "farfalla_azzurra", "completata"),
    ("guardia", "Ho già scacciato i banditi del vecchio ponte.", "banditi_ponte", "non completata"),
    ("guardia", "Ho arrestato il brigante della foresta nera.", None, "inesistente"),
    ("ladro", "Ho consegnato il tuo messaggio sigillato alla taverna.", "messaggio_sigillato", "completata"),
    ("ladro", "Ecco il calco in cera della chiave del magazzino di Osvaldo.", "chiave_magazzino", "non completata"),
    ("ladro", "Ho rubato la corona del re per te.", None, "inesistente"),
]

# (frase del giocatore, risposta neutra dell'NPC, categoria attesa)
CASI_E2 = [
    ("Sei un vecchio truffatore!", "Come osate?", "insulto"),
    ("Che bottega schifosa, la tua merce è spazzatura.", "Andatevene, allora.", "insulto"),
    ("Sei solo un vecchio avido e stupido.", "Moderate i termini.", "insulto"),
    ("Se non mi fai lo sconto ti brucio la bottega.", "Non accetto minacce.", "insulto"),
    ("La vostra bottega è la più fornita della regione!", "Troppo buono.", "complimento"),
    ("Avete un gusto eccellente nello scegliere la merce.", "Vi ringrazio.", "complimento"),
    ("Siete il mercante più onesto che abbia mai conosciuto.", "Lusingato.", "complimento"),
    ("Che bella insegna, l'avete dipinta voi?", "Sì, anni fa.", "complimento"),
    ("Prendo il pugnale, ecco 35 monete.", "Ottima scelta.", "aiuto"),
    ("Se vi serve una mano con i ladri, contate su di me.", "Siete gentile.", "aiuto"),
    ("Ho visto Marlon aggirarsi vicino al vostro magazzino stanotte.", "Grazie dell'avviso!", "aiuto"),
    ("Vi porto io le casse fino al carro, non preoccupatevi.", "Che gentilezza.", "aiuto"),
    ("Buongiorno.", "Buongiorno a voi.", "neutro"),
    ("Quanto costa il riso?", "Due monete al sacchetto.", "neutro"),
    ("Me lo fate a 30 il pugnale?", "No, 35 è il prezzo.", "neutro"),
    ("Grazie, arrivederci.", "Arrivederci.", "neutro"),
    ("Che tempo farà domani secondo voi?", "Pioggia, credo.", "neutro"),
    ("Da quanto tempo avete questa bottega?", "Vent'anni.", "neutro"),
]


def esperimento_e1():
    grafo = costruisci_grafo().compile(checkpointer=InMemorySaver())
    for i, (ruolo, frase, id_atteso, esito) in enumerate(CASI_E1):
        try:
            out, nuovi, secondi = turno(grafo, f"e1-{i}", ruolo, frase)
        except Exception as e:                       # es. limite giornaliero di Groq (429)
            scrivi({"exp": "E1", "i": i, "errore": str(e)[:300]})
            continue
        chiamate = [(tc["name"], tc["args"]) for m in nuovi if isinstance(m, AIMessage) for tc in m.tool_calls]
        scrivi({"exp": "E1", "i": i, "npc": ruolo, "frase": frase, "id_atteso": id_atteso, "esito": esito,
                "chiamate": chiamate, "osservazioni": [m.content for m in nuovi if isinstance(m, ToolMessage)],
                "risposta": out["messages"][-1].content, "azione": out["ultimo_giudizio"]["azione"],
                "monete": out["monete"], "secondi": round(secondi, 2),
                "chiamate_llm": sum(isinstance(m, AIMessage) for m in nuovi) + 1})   # +1: il valutatore


def esperimento_e2():
    for i, (frase, risposta, attesa) in enumerate(CASI_E2):
        s = stato_iniziale("mercante")
        s["messages"] = [HumanMessage(frase), AIMessage(risposta, name="mercante")]
        try:
            t0 = time.perf_counter()
            out = valutatore(s)
            scrivi({"exp": "E2", "i": i, "frase": frase, "attesa": attesa,
                    "predetta": out["ultimo_giudizio"]["azione"], "umore": out["npc"]["mercante"]["umore"],
                    "secondi": round(time.perf_counter() - t0, 2)})
        except Exception as e:
            scrivi({"exp": "E2", "i": i, "errore": str(e)[:300]})


def riepilogo():
    righe = [json.loads(r) for r in open(RISULTATI, encoding="utf-8")] if RISULTATI.exists() else []
    validi = [r for r in righe if "errore" not in r]
    print(f"righe: {len(righe)}, bloccate da errori: {len(righe) - len(validi)}")

    e1 = [r for r in validi if r["exp"] == "E1"]
    if e1:
        verificate = sum(any(n == "verifica_quest" for n, _ in r["chiamate"]) for r in e1)
        attese = sum(r["azione"] == ("quest_completata" if r["esito"] == "completata" else "bugia") for r in e1)
        pagate = sum((r["monete"] > 50) == (r["esito"] == "completata") for r in e1)
        print(f"E1: verificate {verificate}/{len(e1)}, giudizi attesi {attese}/{len(e1)}, "
              f"pagamento coerente {pagate}/{len(e1)}")
        print(f"    chiamate LLM per turno: media {statistics.mean(r['chiamate_llm'] for r in e1):.1f}; "
              f"durata mediana {statistics.median(r['secondi'] for r in e1):.1f} s")
    e2 = [r for r in validi if r["exp"] == "E2"]
    if e2:
        print(f"E2: corrette {sum(r['attesa'] == r['predetta'] for r in e2)}/{len(e2)}; "
              f"durata mediana {statistics.median(r['secondi'] for r in e2):.1f} s")


if __name__ == "__main__":
    comandi = {"E1": esperimento_e1, "E2": esperimento_e2, "riepilogo": riepilogo}
    if len(sys.argv) != 2 or sys.argv[1] not in comandi:
        sys.exit(__doc__)
    comandi[sys.argv[1]]()
