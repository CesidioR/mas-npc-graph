from src.state import WorldState, carica_npc, Evento
import copy

SOGLIA = 10
FATTORE = {"amico": 0.5, "nemico": -0.5}

def gossip(state: WorldState) -> dict: 
    """Propaga ad alleati e rivali l'ultimo giudizio del valutatore: fiducia ed eventi."""
    g = state.get("ultimo_giudizio")
    if not g or abs(g["delta"]) < SOGLIA:
        return {}
    
    ruolo = g['ruolo']
    dati = carica_npc()
    npc = copy.deepcopy(state['npc'])
    eventi = []
       
    for n in npc:
        if n == ruolo:
            continue
        for relazione, fattore in FATTORE.items():
            if ruolo in dati[n][relazione]:
                npc[n]["fiducia"] = max(0, min(100, npc[n]["fiducia"] + int(g["delta"] * fattore)))
                if g["evento"]:  # la fiducia si propaga sempre, la voce solo se c'è un fatto da raccontare
                    eventi.append(Evento(npc=n, testo=g["evento"], fonte="sentito dire"))
    return {"npc": npc, "eventi": eventi}