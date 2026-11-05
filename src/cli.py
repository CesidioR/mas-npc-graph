"""Chat da terminale con gli NPC. Si lancia dalla radice con: python -m src.cli"""

import questionary
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from pathlib import Path

from src.graph import costruisci_grafo
from src.state import carica_npc, stato_iniziale

console = Console()
COLORI = {"mercante": "yellow", "guardia": "blue", "ladro": "magenta"}
COMANDI = "[dim]Comandi: /cambia · /stato · /debug · esci[/]"
# Parole di congedo: dopo la risposta dell'NPC si torna al menu. "ciao" no: è anche un saluto iniziale.
CONGEDI = ("arrivederci", "addio", "a presto", "alla prossima", "devo andare", "me ne vado")
SAVE_DIR = Path(__file__).resolve().parent.parent / "salvataggi"
DB_PATH = SAVE_DIR / "partite.db"
NUOVA = "+ Nuova partita"


def connessione_db() -> SqliteSaver:
    """Checkpointer su file: le partite sopravvivono alla chiusura del programma."""
    SAVE_DIR.mkdir(exist_ok=True)
    return SqliteSaver(sqlite3.connect(DB_PATH, check_same_thread=False))


def scegli_partita(checkpointer: SqliteSaver) -> str | None:
    """thread_id della partita da giocare: una esistente o una nuova. None se annullato."""
    partite = {c.config["configurable"]["thread_id"] for c in checkpointer.list(None)}
    scelta = questionary.select("Scegli la partita", choices=[NUOVA, *sorted(partite)]).ask()
    if scelta == NUOVA:
        nome_p = questionary.text("Nome della nuova partita:").ask()
        while nome_p in partite:
            nome_p = questionary.text("Nome già presente, scegline un altro:").ask()
        return nome_p
    return scelta



def tabella_stato(stato: dict, npc: dict) -> Table:
    """Fiducia e umore di ogni NPC, per il comando /stato."""
    tabella = Table(title="Stato degli NPC")
    tabella.add_column("NPC")
    tabella.add_column("Fiducia", justify="right")
    tabella.add_column("Umore")
    for chiave, s in stato["npc"].items():
        colore = COLORI.get(chiave, "white")
        tabella.add_row(f"[{colore}]{npc[chiave]['nome']}[/]", f"{s['fiducia']}/100", s["umore"])
    return tabella


def mostra_passi(messaggi: list) -> None:
    """Debug: i passi intermedi del ciclo ReAct (tool chiamati e loro risultati)."""
    for m in messaggi:
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                console.print(f"[dim]  → {tc['name']}({escape(str(tc['args']))})[/]")
        elif isinstance(m, ToolMessage):
            console.print(f"[dim]  ← {escape(str(m.content))}[/]")


def scambi_con(messaggi: list, ruolo: str) -> list[tuple[str, str]]:
    """Coppie (frase del giocatore, risposta) dei turni in cui ha risposto l'NPC `ruolo`."""
    scambi, domanda = [], None
    for m in messaggi:
        if isinstance(m, HumanMessage):
            domanda = m.content
        elif isinstance(m, AIMessage) and not m.tool_calls and m.name == ruolo and domanda is not None:
            scambi.append((domanda, m.content))
    return scambi


def mostra_riepilogo(messaggi: list, ruolo: str, nome: str, n: int = 2) -> None:
    """Gli ultimi n scambi con questo NPC, quando torni a parlarci."""
    scambi = scambi_con(messaggi, ruolo)[-n:]
    if scambi:
        console.print(f"[dim]L'ultima volta con {nome}:[/]")
        for domanda, risposta in scambi:
            console.print(f"[dim]  Tu: {escape(domanda)}\n  {nome}: {escape(risposta)}[/]")
        console.print()


def scegli_npc(npc: dict) -> str | None:
    """Ruolo dell'NPC con cui parlare. None se annullato."""
    return questionary.select(
        "Con chi vuoi parlare?",
        choices=[questionary.Choice(f"{info['nome']} ({ruolo})", value=ruolo) for ruolo, info in npc.items()],
    ).ask()


def conversazione(graph, config: dict, ruolo: str, npc: dict, opzioni: dict) -> str:
    """Chat con un NPC finché il giocatore non cambia interlocutore ("cambia") o chiude ("esci")."""
    nome, colore = npc[ruolo]["nome"], COLORI.get(ruolo, "white")
    console.print(f"\nStai parlando con [bold {colore}]{nome}[/]. {COMANDI}\n")
    mostra_riepilogo(graph.get_state(config).values.get("messages", []), ruolo, nome)

    while True:
        testo = Prompt.ask("[bold]Tu[/]").strip()
        comando = testo.lower()
        if comando in {"esci", "quit", "exit"}:
            return "esci"
        if comando == "/cambia":
            return "cambia"
        if not testo:
            continue

        valori = graph.get_state(config).values
        if comando == "/stato":
            console.print(tabella_stato(valori or stato_iniziale(ruolo), npc))
            continue
        if comando == "/debug":
            opzioni["debug"] = not opzioni["debug"]
            console.print(f"[dim]Debug {'attivo' if opzioni['debug'] else 'disattivo'}[/]")
            continue

        messaggio = {"messages": [HumanMessage(content=testo)], "npc_attivo": ruolo}
        if not valori:  # primo turno: la partita non esiste ancora
            messaggio = stato_iniziale(ruolo) | messaggio
        n_prima = len(valori.get("messages", []))

        try:
            with console.status(f"[dim]{nome} ci pensa...[/]"):
                risultato = graph.invoke(messaggio, config)
        except Exception as e:  # es. limite del free tier o rete assente
            console.print(f"[red]Errore: {escape(str(e))}[/]")
            continue

        if opzioni["debug"]:
            mostra_passi(risultato["messages"][n_prima + 1:])  # +1: salta il messaggio del giocatore
        console.print(f"[bold {colore}]{nome}:[/] {escape(risultato['messages'][-1].content)}\n")

        if any(parola in comando for parola in CONGEDI):  # l'NPC ha già risposto al saluto
            return "cambia"

  
def main():
    memory = connessione_db()
    graph = costruisci_grafo().compile(checkpointer=memory)
    npc = carica_npc()

    console.print(Panel.fit("[bold]Benvenuto al villaggio[/]", border_style="green"))

    thread_id = scegli_partita(memory)
    if not thread_id:
        return
    config = {"configurable": {"thread_id": thread_id}}
    if graph.get_state(config).values:
        console.print(f"\n[bold]Bentornato nella partita \"{escape(thread_id)}\".[/]")

    opzioni = {"debug": False}
    try:
        while True:
            ruolo = scegli_npc(npc)
            if ruolo is None or conversazione(graph, config, ruolo, npc, opzioni) == "esci":
                break
    except (KeyboardInterrupt, EOFError):
        pass
    console.print("\n[dim]Alla prossima, viaggiatore.[/]")


if __name__ == "__main__":
    main()
