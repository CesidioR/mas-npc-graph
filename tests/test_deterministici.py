"""Test senza LLM: gossip, sensibilità, clamp, visibilità, trascrizione, soglie di sconto e furto."""
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.nodes.gossip import gossip
from src.nodes.npc_agent import messaggi_visibili
from src.nodes.valutatore import DELTA_FIDUCIA, trascrivi_turno
from src.prompts import system_prompt
from src.state import TETTO_PAROLE, aggiorna_fiducia, carica_npc, delta_personale, stato_iniziale
from src.tools import verifica_quest


def giudizio(ruolo, azione, delta, evento=None):
    return {"ruolo": ruolo, "azione": azione, "delta": delta, "evento": evento}


def fiducie(npc):
    return {r: s["fiducia"] for r, s in npc.items()}


# --- stato e tool ---

def test_stato_iniziale_dai_json():
    s = stato_iniziale()
    assert fiducie(s["npc"]) == {"mercante": 60, "guardia": 50, "ladro": 30}
    assert s["messages"] == [] and s["eventi"] == [] and s["ultimo_giudizio"] is None


@pytest.mark.parametrize("id_quest, atteso", [
    ("cassa_spezie", "completata. Ricompensa: 30"),
    ("pellicce_lupo", "non completata"),
    ("drago_montagna", "inesistente. Quest note:"),
])
def test_verifica_quest(id_quest, atteso):
    assert atteso in verifica_quest.invoke({"id_quest": id_quest})


# --- sensibilità ---

@pytest.mark.parametrize("ruolo, azione, atteso", [
    ("mercante", "insulto", -10), ("mercante", "quest_completata", 15),
    ("guardia", "insulto", -6), ("guardia", "quest_completata", 22),     # credulone
    ("ladro", "insulto", -12), ("ladro", "quest_completata", 7),         # diffidente
    ("ladro", "neutro", 0),
])
def test_delta_personale(ruolo, azione, atteso):
    assert delta_personale(DELTA_FIDUCIA[azione], carica_npc()[ruolo]) == atteso


def test_delta_personale_senza_sensibilita():
    assert delta_personale(-15, {}) == -15


# --- tetto della fiducia "a parole" ---

@pytest.mark.parametrize("fiducia, delta, azione, attesa", [
    (60, 8, "aiuto", 65),                # le parole si fermano al tetto
    (64, 5, "complimento", 65),
    (60, 15, "quest_completata", 75),    # le quest lo superano
    (80, 5, "complimento", 80),          # chi è già sopra non viene abbassato
    (70, -10, "insulto", 60),            # le azioni negative non c'entrano
    (95, 15, "quest_completata", 100),   # il clamp resta
])
def test_aggiorna_fiducia(fiducia, delta, azione, attesa):
    assert aggiorna_fiducia(fiducia, delta, azione) == attesa


def test_tetto_sotto_le_soglie_di_sconto_e_furto():
    npc = carica_npc()
    assert TETTO_PAROLE < npc["mercante"]["soglia_sconto"] and TETTO_PAROLE < npc["ladro"]["soglia_furto"]


def test_gossip_rispetta_il_tetto():
    s = stato_iniziale()
    s["npc"]["guardia"]["fiducia"] = 62
    s["ultimo_giudizio"] = giudizio("mercante", "aiuto", 8)      # alleato: +4 x 1.5 = +6, ma tetto 65
    assert gossip(s)["npc"]["guardia"]["fiducia"] == 65


# --- gossip ---

def test_gossip_insulto_a_osvaldo():
    s = stato_iniziale()
    s["ultimo_giudizio"] = giudizio("mercante", "insulto", -15, "Il giocatore ha insultato Osvaldo.")
    out = gossip(s)
    # alleato: -7.5 x 0.6 = -4 ; rivale: +7.5 x 0.5 = +3
    assert fiducie(out["npc"]) == {"mercante": 60, "guardia": 46, "ladro": 33}
    assert sorted(e["npc"] for e in out["eventi"]) == ["guardia", "ladro"]
    assert all(e["fonte"] == "sentito dire" for e in out["eventi"])
    assert s["npc"]["guardia"]["fiducia"] == 50          # lo stato di partenza non viene modificato


def test_gossip_quest_senza_evento():
    s = stato_iniziale()
    s["ultimo_giudizio"] = giudizio("mercante", "quest_completata", 15)
    out = gossip(s)
    # alleato: +7.5 x 1.5 = +11 ; rivale: -7.5 x 1.2 = -9
    assert fiducie(out["npc"]) == {"mercante": 60, "guardia": 61, "ladro": 21}
    assert out["eventi"] == []


@pytest.mark.parametrize("g", [None, giudizio("mercante", "neutro", 0)])
def test_gossip_niente_da_propagare(g):
    s = stato_iniziale()
    s["ultimo_giudizio"] = g
    assert gossip(s) == {}


def test_gossip_clamp():
    s = stato_iniziale()
    s["npc"]["guardia"]["fiducia"], s["npc"]["ladro"]["fiducia"] = 98, 2
    s["ultimo_giudizio"] = giudizio("mercante", "quest_completata", 15)
    out = gossip(s)
    assert out["npc"]["guardia"]["fiducia"] == 100 and out["npc"]["ladro"]["fiducia"] == 0


# --- visibilità e trascrizione ---

def _tool_call(i):
    return [{"name": "verifica_quest", "args": {"id_quest": "cassa_spezie"}, "id": i}]


STORIA = [
    HumanMessage("a"), AIMessage("", tool_calls=_tool_call("t1"), name="mercante"),
    ToolMessage("ok", tool_call_id="t1"), AIMessage("b", name="mercante"),
    HumanMessage("c"), AIMessage("d", name="guardia"),
    HumanMessage("e"),
]


def contenuti(ms):
    return [m.content for m in ms]


@pytest.mark.parametrize("ruolo, atteso", [
    ("guardia", ["c", "d", "e"]),
    ("mercante", ["a", "", "ok", "b", "e"]),     # tool call e risultato restano insieme
    ("ladro", ["e"]),
])
def test_messaggi_visibili(ruolo, atteso):
    assert contenuti(messaggi_visibili(STORIA, ruolo)) == atteso


def test_messaggi_visibili_a_meta_ciclo_react():
    meta = STORIA + [AIMessage("", tool_calls=_tool_call("t2"), name="guardia"), ToolMessage("r", tool_call_id="t2")]
    assert contenuti(messaggi_visibili(meta, "guardia"))[-3:] == ["e", "", "r"]


def test_messaggi_visibili_cronologia_vuota():
    assert messaggi_visibili([], "guardia") == []


def test_trascrivi_turno():
    assert trascrivi_turno(STORIA[:4], "Osvaldo") == "Giocatore: a\nVerifica nei registri: ok\nOsvaldo: b"
    assert trascrivi_turno(STORIA, "Patrizio") == "Giocatore: e"      # solo il turno corrente


# --- soglie di sconto e furto nel prompt ---

def prompt_con_fiducia(ruolo, fiducia):
    s = stato_iniziale(ruolo)
    s["npc"][ruolo]["fiducia"] = fiducia
    return system_prompt(ruolo, s)


@pytest.mark.parametrize("fiducia, atteso", [(74, "Non fare sconti"), (75, "sconto fino al 20%")])
def test_soglia_sconto(fiducia, atteso):
    assert atteso in prompt_con_fiducia("mercante", fiducia)


@pytest.mark.parametrize("fiducia, atteso", [(69, "rifiuta"), (70, "puoi accettare il lavoro")])
def test_soglia_furto(fiducia, atteso):
    p = prompt_con_fiducia("ladro", fiducia)
    riga = next(r for r in p.splitlines() if r.startswith("Se il giocatore ti chiede di rubare"))
    assert atteso in riga


def test_regole_solo_per_chi_le_ha():
    p = prompt_con_fiducia("guardia", 90)
    assert "sconto" not in p and "ti chiede di rubare" not in p


# --- economia: vendite, ricompense, inventario ---

from langgraph.types import Command

from src.tools import accetta_furto, consulta_inventario, prezzo_minimo, riscuoti_quest, vendi


def esegui(strumento, s, **args):
    """Chiama un tool come farebbe il ToolNode; restituisce l'aggiornamento dello stato o il testo di rifiuto."""
    r = strumento.invoke({"type": "tool_call", "id": "t1", "name": strumento.name, "args": {**args, "state": s}})
    return r.update if isinstance(r, Command) else r.content


@pytest.mark.parametrize("prezzo, fiducia, atteso", [(35, 60, 35), (35, 75, 28), (6, 80, 5), (120, 74, 120)])
def test_prezzo_minimo(prezzo, fiducia, atteso):
    assert prezzo_minimo(prezzo, fiducia, carica_npc()["mercante"]) == atteso


def test_vendita_riuscita():
    s = stato_iniziale()
    up = esegui(vendi, s, oggetto="pugnale", prezzo=35)
    assert up["monete"] == 15 and up["inventario"]["pugnale"] == 2 and up["inventario"]["armatura"] == 1
    assert s["monete"] == 50 and s["inventario"]["pugnale"] == 3       # stato di partenza intatto


@pytest.mark.parametrize("oggetto, prezzo, fiducia, motivo", [
    ("pugnale", 28, 60, "troppo poche"),      # sconto non sbloccato
    ("pugnale", 27, 80, "troppo poche"),      # oltre lo sconto massimo
    ("armatura", 120, 60, "solo 50 monete"),  # borsellino insufficiente
    ("spada", 10, 60, "non è in vendita"),
])
def test_vendita_rifiutata(oggetto, prezzo, fiducia, motivo):
    s = stato_iniziale()
    s["npc"]["mercante"]["fiducia"] = fiducia
    assert motivo in esegui(vendi, s, oggetto=oggetto, prezzo=prezzo)


def test_vendita_scontata_sopra_soglia():
    s = stato_iniziale()
    s["npc"]["mercante"]["fiducia"] = 80
    assert esegui(vendi, s, oggetto="pugnale", prezzo=28)["monete"] == 22


def test_oggetto_esaurito():
    s = stato_iniziale()
    s["monete"], s["inventario"]["armatura"] = 200, 0
    assert "esaurito" in esegui(vendi, s, oggetto="armatura", prezzo=120)


def test_consulta_inventario_legge_lo_stato():
    s = stato_iniziale()
    s["inventario"]["pugnale"] = 1
    inv = consulta_inventario.invoke({"type": "tool_call", "id": "t1", "name": "consulta_inventario",
                                      "args": {"state": s}}).content
    assert '"pugnale": {"prezzo": 35, "quantita": 1}' in inv


def test_ricompensa_pagata():
    s = stato_iniziale("mercante")
    up = esegui(riscuoti_quest, s, id_quest="cassa_spezie")
    assert up["monete"] == 80 and up["quest_riscosse"] == ["cassa_spezie"]


@pytest.mark.parametrize("ruolo, id_quest, riscosse, motivo", [
    ("mercante", "cassa_spezie", ["cassa_spezie"], "già stata pagata"),
    ("mercante", "pellicce_lupo", [], "non è completata"),
    ("mercante", "farfalla_azzurra", [], "non l'hai assegnata tu"),
    ("guardia", "drago_montagna", [], "inesistente"),
])
def test_ricompensa_rifiutata(ruolo, id_quest, riscosse, motivo):
    s = stato_iniziale(ruolo)
    s["quest_riscosse"] = riscosse
    assert motivo in esegui(riscuoti_quest, s, id_quest=id_quest)


def test_lo_stato_iniettato_non_e_visibile_all_llm():
    for t in (vendi, riscuoti_quest, consulta_inventario, accetta_furto):
        assert "state" not in t.tool_call_schema.model_json_schema().get("properties", {})


# --- furto su commissione (Marlon, soglia 70, compenso minimo 40% del prezzo) ---

def stato_ladro(fiducia=75, monete=50):
    s = stato_iniziale("ladro")
    s["npc"]["ladro"]["fiducia"], s["monete"] = fiducia, monete
    return s


def test_furto_riuscito():
    s = stato_ladro()
    up = esegui(accetta_furto, s, oggetto="pugnale", compenso=14)      # 35 x 0.4 = 14
    assert up["monete"] == 36 and up["inventario"]["pugnale"] == 2      # sparisce dalla bottega di Osvaldo
    assert s["inventario"]["pugnale"] == 3                              # stato di partenza intatto


@pytest.mark.parametrize("fiducia, oggetto, compenso, monete, motivo", [
    (69, "pugnale", 20, 50, "Non ti fidi abbastanza"),     # sotto la soglia
    (75, "pugnale", 13, 50, "troppo poche"),               # sotto il compenso minimo
    (75, "pugnale", -5, 50, "troppo poche"),               # un compenso negativo darebbe monete al giocatore
    (75, "pugnale", 36, 50, "più di quanto costa"),        # oltre il prezzo pieno (35): conviene comprarlo
    (75, "spada", 20, 50, "non vende"),
    (75, "armatura", 48, 30, "solo 30 monete"),            # borsellino insufficiente
])
def test_furto_rifiutato(fiducia, oggetto, compenso, monete, motivo):
    assert motivo in esegui(accetta_furto, stato_ladro(fiducia, monete), oggetto=oggetto, compenso=compenso)


def test_furto_oggetto_esaurito():
    s = stato_ladro()
    s["inventario"]["armatura"] = 0
    assert "non c'è più" in esegui(accetta_furto, s, oggetto="armatura", compenso=60)


def test_marlon_conosce_la_merce_della_bottega():
    p = system_prompt("ladro", stato_iniziale("ladro"))
    assert "Nella bottega di Osvaldo ci sono: armatura, pugnale" in p
    assert "Nella bottega di Osvaldo" not in system_prompt("guardia", stato_iniziale("guardia"))


def test_furto_al_prezzo_pieno_accettato():
    assert esegui(accetta_furto, stato_ladro(), oggetto="pugnale", compenso=35)["monete"] == 15


# --- il bonus di una quest richiede un pagamento nel turno corrente ---

from src.nodes.valutatore import ricompensa_pagata


def test_ricompensa_pagata_nel_turno():
    pagata = ToolMessage("Pagate 30 monete.", tool_call_id="t1", artifact={"quest_pagata": "cassa_spezie"})
    rifiutata = ToolMessage("Già pagata: nessun pagamento.", tool_call_id="t2")
    assert ricompensa_pagata([HumanMessage("a"), pagata])
    assert not ricompensa_pagata([HumanMessage("a"), rifiutata])
    assert not ricompensa_pagata([HumanMessage("a"), pagata, HumanMessage("b")])   # pagamento di un turno precedente
