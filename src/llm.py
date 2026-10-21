"""Punto unico di creazione degli LLM del progetto.

Tutti i nodi del grafo prendono il modello da qui con get_llm(ruolo):
provider e modello si scelgono nel .env, senza toccare il codice.
"""

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

load_dotenv()

# Temperatura per ruolo: bassa dove serve precisione, più alta per il dialogo.
TEMPERATURE = {
    "router": 0.0,      # deve solo scegliere l'NPC giusto
    "valutatore": 0.1,  # produce numeri (Δfiducia, umore): servono coerenza e ripetibilità
    "dialogo": 0.7,     # le battute degli NPC: un po' di varietà
}

# Opzioni specifiche di alcuni provider.
PROVIDER_EXTRA = {
    "google_genai": {"thinking_level": "low"},  # Gemini 3.x: riduce la latenza
}


@lru_cache  # un'istanza per ruolo, creata una sola volta
def get_llm(ruolo: str = "dialogo") -> BaseChatModel:
    if ruolo not in TEMPERATURE:
        raise ValueError(f"Ruolo sconosciuto: {ruolo!r}. Validi: {list(TEMPERATURE)}")

    provider = os.getenv("LLM_PROVIDER")
    model = os.getenv("LLM_MODEL")
    if not provider or not model:
        raise RuntimeError("Imposta LLM_PROVIDER e LLM_MODEL nel file .env")

    return init_chat_model(
        model,
        model_provider=provider,
        temperature=TEMPERATURE[ruolo],
        timeout=60,
        max_retries=2,
        **PROVIDER_EXTRA.get(provider, {}),
    )