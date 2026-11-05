from langchain_core.tools import tool
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
def consulta_inventario() -> dict:
    """Controlla l'inventario del mercante. 
    Usalo quando un giocatore chiede cosa vendi o quanto costa un oggetto."""
    npc = carica_npc()
    inventario = npc["mercante"]["inventario"]
    return inventario
    

TOOLS = [verifica_quest, consulta_inventario]  # tutti: li esegue il ToolNode
_PER_NOME = {t.name: t for t in TOOLS}


def strumenti_npc(ruolo: str) -> list:
    """I tool che l'NPC può usare, dal campo 'strumenti' di npc.json (per bind_tools)."""
    return [_PER_NOME[nome_tool] for nome_tool in carica_npc()[ruolo].get("strumenti", [])]
