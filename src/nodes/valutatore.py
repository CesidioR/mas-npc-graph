from typing import Literal, get_args
from src.state import WorldState, Umore, aggiorna_fiducia, carica_npc, delta_personale, Evento, Giudizio
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from pydantic import BaseModel, Field
from src.llm import get_llm


Azione = Literal["insulto", "bugia", "quest_completata", "aiuto", "complimento", "neutro"]


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
    "insulto": -10,
    "bugia": -20,
    "quest_completata": 15,
    "aiuto": 8,
    "complimento": 5,
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
- aiuto: fa qualcosa di concreto per l'NPC: compra la sua merce, offre il suo aiuto, gli fa un favore o gli dà un'informazione utile. Es: "Prendo il pugnale.", "Se vi serve una mano con i banditi, ci sono io."
- complimento: è esplicitamente gentile, cortese o lusinghiero. Es: "La vostra bottega è la più fornita del villaggio." Un semplice saluto o un "grazie" sono neutri.
- neutro: tutto il resto: saluti, domande, richieste di prezzi, contrattazioni senza acquisto, chiacchiere.
Se nel turno ci sono più cose, scegli la più grave: bugia e insulto prevalgono sulle altre.

2. umore: l'umore dell'NPC DOPO questo turno. Parti dall'umore attuale e cambialo solo se il turno lo giustifica; un turno neutro di solito lo lascia invariato.

3. evento: una sola frase in terza persona, con il nome dell'NPC, che descriva un fatto da ricordare in futuro. Esempi:
- "Il giocatore ha mentito a Osvaldo sulle pellicce di lupo."
- "Il giocatore ha insultato Osvaldo chiamandolo truffatore."
- "Il giocatore ha recuperato la cassa di spezie per Osvaldo."
Restituisci null se il turno è neutro o non c'è niente di significativo da ricordare.

Giudica solo ciò che fa il giocatore, non le risposte dell'NPC."""

def turno_corrente(messages: list) -> list:
    """I messaggi del turno corrente: dall'ultimo messaggio del giocatore in poi."""
    inizio = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
    return messages[inizio:]


def ricompensa_pagata(messages: list) -> bool:
    """True se nel turno corrente riscuoti_quest ha davvero pagato una ricompensa."""
    return any(isinstance(m, ToolMessage) and isinstance(m.artifact, dict) and m.artifact.get("quest_pagata")
               for m in turno_corrente(messages))


def trascrivi_turno(messages: list, nome_npc: str) -> str:
    """Il turno corrente (dall'ultimo messaggio del giocatore) come testo da far leggere al giudice."""
    righe = []
    for m in turno_corrente(messages):
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
    azione, testo_evento = v.azione, v.evento
    # Una quest vale fiducia una sola volta: nel turno in cui la ricompensa viene pagata.
    # Senza pagamento (già riscossa, quest di un altro NPC, NPC che non ha pagato) il turno è neutro.
    if azione == "quest_completata" and not ricompensa_pagata(state["messages"]):
        azione, testo_evento = "neutro", None
    evento = Evento(npc=ruolo, testo=testo_evento, fonte="vissuto") if testo_evento else None

    delta = delta_personale(DELTA_FIDUCIA[azione], carica_npc()[ruolo])  # il carattere dell'NPC
    nuova_fiducia = aggiorna_fiducia(attuale["fiducia"], delta, azione)

    npc = {**state["npc"], ruolo: {"fiducia": nuova_fiducia, "umore": v.umore}}
    return {"npc": npc, "eventi": [evento] if evento else [],
            "ultimo_giudizio": Giudizio(ruolo=ruolo, azione=azione, delta=DELTA_FIDUCIA[azione], evento=testo_evento)}
    
