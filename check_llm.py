"""Verifica che i provider LLM configurati nel .env rispondano.

Uso:  python check_llm.py
"""

import os
import time

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

# (provider per LangChain, variabile della chiave, variabile del modello, opzioni extra)
PROVIDERS = [
    ("groq", "GROQ_API_KEY", "GROQ_MODEL", {}),
    ("google_genai", "GEMINI_API_KEY", "GEMINI_MODEL", {"thinking_level": "low"}),
]

TIMEOUT_S = 30


def check(provider: str, key_var: str, model_var: str, extra: dict) -> None:
    if not os.getenv(key_var):
        print(f"[{provider}] SALTATO: manca {key_var} nel .env")
        return

    model = os.getenv(model_var)
    if not model:
        print(f"[{provider}] SALTATO: manca {model_var} nel .env")
        return

    print(f"[{provider}] chiamo {model} ...", flush=True)
    start = time.time()
    try:
        llm = init_chat_model(
            model,
            model_provider=provider,
            timeout=TIMEOUT_S,   # non resta mai appeso più di così
            max_retries=0,       # niente retry silenziosi: l'errore emerge subito
            **extra,
        )
        reply = llm.invoke("Rispondi solo: ok")
        print(f"[{provider}] OK in {time.time() - start:.1f}s → {reply.text!r}")
    except Exception as e:  # vogliamo vedere qualunque errore, ma testare anche gli altri provider
        print(f"[{provider}] ERRORE dopo {time.time() - start:.1f}s → {type(e).__name__}: {e}")


if __name__ == "__main__":
    for provider, key_var, model_var, extra in PROVIDERS:
        check(provider, key_var, model_var, extra)