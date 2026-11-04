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
COMANDI = "[dim]Comandi: /stato · /debug · esci[/]"
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


def mostra_riepilogo(messaggi: list, npc: dict, n: int = 4) -> None:
    """Le ultime n battute della partita, alla ripresa (senza tool call e risultati)."""
    battute = [m for m in messaggi
               if isinstance(m, HumanMessage) or (isinstance(m, AIMessage) and not m.tool_calls)]
    for m in battute[-n:]:
        if isinstance(m, HumanMessage):
            console.print(f"[dim]Tu: {escape(m.content)}[/]")
        else:
            chi = npc.get(m.name, {}).get("nome", "?")  # battute vecchie senza firma → "?"
            console.print(f"[dim]{chi}: {escape(m.content)}[/]")


def main():
    memory = connessione_db()
    graph = costruisci_grafo().compile(checkpointer=memory)
    npc = carica_npc()

    console.print(Panel.fit("[bold]Benvenuto al villaggio[/]", border_style="green"))
    
    thread_id = scegli_partita(memory)
    if not thread_id:
        return
    config = {"configurable": {"thread_id": thread_id}}

    salvato = graph.get_state(config).values
    if salvato:
        console.print(f"\n[bold]Partita \"{escape(thread_id)}\": eri rimasto qui[/]")
        mostra_riepilogo(salvato["messages"], npc)

    # Fino alla Fase 5 (router) l'NPC si sceglie dal menu.
    scelta = questionary.select(
        "Con chi vuoi parlare?",
        choices=[questionary.Choice(f"{info['nome']} ({chiave})", value=chiave) for chiave, info in npc.items()],
    ).ask()
    if scelta is None:  # Ctrl+C nel menu
        return
    nome, colore = npc[scelta]["nome"], COLORI.get(scelta, "white")
    console.print(f"\nStai parlando con [bold {colore}]{nome}[/]. {COMANDI}\n")

    debug = False
    try:
        while True:
            testo = Prompt.ask("[bold]Tu[/]").strip()
            comando = testo.lower()
            if comando in {"esci", "quit", "exit"}:
                break
            if not testo:
                continue

            valori = graph.get_state(config).values
            if comando == "/stato":
                console.print(tabella_stato(valori or stato_iniziale(scelta), npc))
                continue
            if comando == "/debug":
                debug = not debug
                console.print(f"[dim]Debug {'attivo' if debug else 'disattivo'}[/]")
                continue

            messaggio = {"messages": [HumanMessage(content=testo)], "npc_attivo": scelta}
            if not valori:  # primo turno: la partita non esiste ancora
                messaggio = stato_iniziale(scelta) | messaggio
            n_prima = len(valori.get("messages", []))

            try:
                with console.status(f"[dim]{nome} ci pensa...[/]"):
                    risultato = graph.invoke(messaggio, config)
            except Exception as e:  # es. limite del free tier o rete assente
                console.print(f"[red]Errore: {escape(str(e))}[/]")
                continue

            if debug:
                mostra_passi(risultato["messages"][n_prima + 1:])  # +1: salta il messaggio del giocatore
            console.print(f"[bold {colore}]{nome}:[/] {escape(risultato['messages'][-1].content)}\n")
    except (KeyboardInterrupt, EOFError):
        pass
    console.print("\n[dim]Alla prossima, viaggiatore.[/]")


if __name__ == "__main__":
    main()
