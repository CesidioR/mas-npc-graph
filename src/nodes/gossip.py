from src.state import WorldState, aggiorna_fiducia, carica_npc, delta_personale, Evento
import copy

FATTORE = {"amico": 0.5, "nemico": -0.5}

def gossip(state: WorldState) -> dict: 
    """Propaga ad alleati e rivali l'ultimo giudizio del valutatore: fiducia ed eventi."""
    g = state.get("ultimo_giudizio")
    if not g or g["delta"] == 0:  # nessuna soglia: si propaga ogni azione che cambia la fiducia
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
                delta = delta_personale(g["delta"] * fattore, dati[n])
                npc[n]["fiducia"] = aggiorna_fiducia(npc[n]["fiducia"], delta, g["azione"])
                if g["evento"]:  # la fiducia si propaga sempre, la voce solo se c'è un fatto da raccontare
                    eventi.append(Evento(npc=n, testo=g["evento"], fonte="sentito dire"))
    return {"npc": npc, "eventi": eventi}