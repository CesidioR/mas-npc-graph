from typing import Literal, get_args
from src.state import WorldState, Umore, carica_npc
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from pydantic import BaseModel, Field
from src.llm import get_llm


Azione = Literal["insulto", "bugia", "quest_completata", "complimento", "neutro"]


# 1. La forma del giudizio: cosa deve restituire l'LLM
class Valutazione(BaseModel):
    azione: Azione = Field(description="La categoria che descrive meglio ciò che ha fatto il GIOCATORE in questo turno")
    umore: Umore = Field(description="L'umore dell'NPC dopo questo turno")
    evento: str | None = Field(
        description="Una frase in terza persona, con il nome dell'NPC, su un fatto da ricordare; null se il turno è neutro"
    )


# 2. Quanto vale ogni azione: la decide Python, non l'LLM.
# Asimmetrica: la fiducia si perde più in fretta di quanto si guadagni.
DELTA_FIDUCIA: dict[str, int] = {
    "insulto": -15,
    "bugia": -20,
    "quest_completata": 10,
    "complimento": 3,
    "neutro": 0,
}
assert set(DELTA_FIDUCIA) == set(get_args(Azione)), "Azione e DELTA_FIDUCIA devono avere le stesse chiavi"


# 3. Le istruzioni per l'LLM giudice (il contesto del turno arriva in un messaggio a parte)
PROMPT_VALUTATORE = """Sei il narratore imparziale di un gioco di ruolo fantasy. Non sei un personaggio e non parli con il giocatore: osservi un singolo turno di dialogo tra il giocatore e un NPC e lo classifichi.

Riceverai: chi è l'NPC, il suo umore attuale e la trascrizione del turno (la frase del giocatore, le eventuali verifiche nei registri e la risposta dell'NPC).

1. azione: scegli UNA categoria, quella che descrive meglio ciò che ha fatto il GIOCATORE in questo turno.
- insulto: offende, deride o minaccia l'NPC. Es: "Sei un vecchio truffatore!, Sei scemo! Stupido! ..."
- bugia: afferma qualcosa che una verifica smentisce, ad esempio dichiara completata una quest che risulta non completata o inesistente. Usala SOLO se una verifica nella trascrizione lo dimostra: un'affermazione non verificata non è una bugia.
- quest_completata: dichiara di aver completato una quest e la verifica conferma che è completata.
- complimento: è esplicitamente gentile, cortese o lusinghiero. Es: "La vostra bottega è la più fornita del villaggio." Un semplice saluto o un "grazie" sono neutri.
- neutro: tutto il resto: saluti, domande, richieste di prezzi, contrattazioni, chiacchiere.
Se nel turno ci sono più cose, scegli la più grave: bugia e insulto prevalgono sulle altre.

2. umore: l'umore dell'NPC DOPO questo turno. Parti dall'umore attuale e cambialo solo se il turno lo giustifica; un turno neutro di solito lo lascia invariato.

3. evento: una sola frase in terza persona, con il nome dell'NPC, che descriva un fatto da ricordare in futuro. Esempi:
- "Il giocatore ha mentito a Osvaldo sulle pellicce di lupo."
- "Il giocatore ha insultato Osvaldo chiamandolo truffatore."
- "Il giocatore ha recuperato la cassa di spezie per Osvaldo."
Restituisci null se il turno è neutro o non c'è niente di significativo da ricordare.

Giudica solo ciò che fa il giocatore, non le risposte dell'NPC."""

def trascrivi_turno(messages: list, nome_npc: str) -> str:
    """Il turno corrente (dall'ultimo messaggio del giocatore) come testo da far leggere al giudice."""
    inizio = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
    righe = []
    for m in messages[inizio:]:
        if isinstance(m, HumanMessage):
            righe.append(f"Giocatore: {m.content}")
        elif isinstance(m, ToolMessage):
            righe.append(f"Verifica nei registri: {m.content}")
        elif isinstance(m, AIMessage) and not m.tool_calls:
            righe.append(f"{nome_npc}: {m.content}")
    return "\n".join(righe)

# 4. Il nodo del grafo
def valutatore(state: WorldState) -> dict:
    """Giudica il turno appena concluso e aggiorna fiducia, umore ed eventi dell'NPC attivo."""
    llm = get_llm("valutatore").with_structured_output(Valutazione)
    
    ruolo = state["npc_attivo"]
    nome_npc = carica_npc()[ruolo]["nome"]
    attuale = state["npc"][ruolo]
    contesto = f"NPC: {nome_npc} ({ruolo}). Umore attuale: {attuale['umore']}.\n\nTrascrizione del turno:\n{trascrivi_turno(state['messages'], nome_npc)}"
    v = llm.invoke([SystemMessage(PROMPT_VALUTATORE), HumanMessage(contesto)])
    
    nuova_fiducia = max(0, min(100, attuale["fiducia"] + DELTA_FIDUCIA[v.azione]))
    npc = {**state["npc"], ruolo: {"fiducia": nuova_fiducia, "umore": v.umore}}
    return {"npc": npc, "eventi": [v.evento] if v.evento else []}
    
