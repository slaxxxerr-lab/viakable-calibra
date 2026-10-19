"""Pluggable LLM providers for Asistente Calibra. No Streamlit dependency.

Provider order: Ollama (local) → Gemini (free cloud) → Groq (free cloud).
Each provider is tried in sequence; the first that succeeds returns the answer.

secret_fn(section, key) reads a key from .streamlit/secrets.toml (passed in from app.py).
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Callable

import requests

_TIMEOUT = 30
_OLLAMA_URL = "http://localhost:11434"
_OLLAMA_MODELS = ["qwen2.5:3b", "qwen2.5:1.5b", "llama3.2:3b", "llama3.2:1b", "llama3.2"]
_GEMINI_DEFAULT_MODEL = "gemini-3.8-flash"
_GROQ_DEFAULT_MODEL = "llama-3.1-8b-instant"

SecretFn = Callable[[str, str], str | None]


class LLMUnavailable(Exception):
    """Raised when no provider can respond."""


@dataclass
class Provider:
    label: str
    model: str | None
    ok: bool
    note: str
    local: bool = False


def _ollama_model() -> str | None:
    """Return the first available Ollama model, or None."""
    try:
        r = requests.get(f"{_OLLAMA_URL}/api/tags", timeout=4)
        r.raise_for_status()
        names = [m["name"] for m in r.json().get("models", [])]
        for preferred in _OLLAMA_MODELS:
            for name in names:
                if name.startswith(preferred.split(":")[0]):
                    return name
        return names[0] if names else None
    except Exception:
        return None


def _ollama_available() -> tuple[bool, str, str]:
    """(ok, model_name, note)"""
    try:
        requests.get(f"{_OLLAMA_URL}/api/tags", timeout=4).raise_for_status()
    except Exception:
        return False, "", "Ollama no responde: instálalo y ábrelo (ver instrucciones)"
    model = _ollama_model()
    if not model:
        return False, "", "Ollama corre, pero falta un modelo: ejecuta ollama pull qwen2.5:3b"
    return True, model, ""


def _gemini_model(secret_fn: SecretFn) -> str:
    configured = secret_fn("gemini", "model") if secret_fn else None
    return configured or _GEMINI_DEFAULT_MODEL


def _groq_model(secret_fn: SecretFn) -> str:
    configured = secret_fn("groq", "model") if secret_fn else None
    return configured or _GROQ_DEFAULT_MODEL


def _ollama_chat(model: str, sysmsg: str, messages: list[dict], max_tokens: int) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": sysmsg}] + messages,
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": 0.3},
    }
    r = requests.post(f"{_OLLAMA_URL}/api/chat", json=payload, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def _gemini_chat(api_key: str, model: str, sysmsg: str, messages: list[dict], max_tokens: int) -> str:
    url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent"
    parts = [{"text": f"[SYSTEM]\n{sysmsg}\n[/SYSTEM]\n"}]
    for m in messages:
        parts.append({"text": f"[{m['role'].upper()}]\n{m['content']}\n"})
    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.3},
    }
    for attempt in range(3):
        r = requests.post(url, json=payload, params={"key": api_key}, timeout=_TIMEOUT)
        if r.status_code in (429, 503):
            time.sleep(2 * (attempt + 1))
            continue
        break
    if not r.ok:
        raise LLMUnavailable(f"Gemini HTTP {r.status_code}")
    candidates = r.json().get("candidates", [])
    if not candidates:
        raise LLMUnavailable("Gemini returned no candidates")
    parts_out = candidates[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts_out).strip()


def _groq_chat(api_key: str, model: str, sysmsg: str, messages: list[dict], max_tokens: int) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": sysmsg}] + messages,
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        json=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=_TIMEOUT,
    )
    if r.status_code == 429:
        time.sleep(2)
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=_TIMEOUT,
        )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def providers(secret_fn: SecretFn | None = None) -> list[Provider]:
    """Return status of all providers."""
    result = []

    ol_ok, ol_model, ol_note = _ollama_available()
    result.append(Provider(label="Ollama", model=ol_model or None, ok=ol_ok, note=ol_note, local=True))

    gem_key = secret_fn("gemini", "api_key") if secret_fn else None
    if gem_key:
        result.append(Provider(label="Gemini", model=_gemini_model(secret_fn), ok=True, note="", local=False))
    else:
        result.append(Provider(label="Gemini", model=None, ok=False, note="sin llave en secrets.toml", local=False))

    groq_key = secret_fn("groq", "api_key") if secret_fn else None
    if groq_key:
        result.append(Provider(label="Groq", model=_groq_model(secret_fn), ok=True, note="", local=False))
    else:
        result.append(Provider(label="Groq", model=None, ok=False, note="sin llave en secrets.toml", local=False))

    return result


def any_available(secret_fn: SecretFn | None = None) -> bool:
    return any(p.ok for p in providers(secret_fn))


def chat(
    sysmsg: str,
    messages: list[dict],
    secret_fn: SecretFn | None = None,
    max_tokens: int = 360,
    allow: tuple = ("ollama", "gemini", "groq"),
) -> dict:
    """Generate a response using the first available allowed provider.

    Returns {"text": str, "provider": str, "model": str, "local": bool}.
    Raises LLMUnavailable if all providers fail or are not allowed.
    """
    errors = []

    if "ollama" in allow:
        ol_ok, ol_model, _ = _ollama_available()
        if ol_ok:
            try:
                text = _ollama_chat(ol_model, sysmsg, messages, max_tokens)
                return {"text": text, "provider": "Ollama", "model": ol_model, "local": True}
            except Exception as ex:
                errors.append(f"Ollama: {ex}")

    if "gemini" in allow:
        gem_key = secret_fn("gemini", "api_key") if secret_fn else None
        if gem_key:
            try:
                model = _gemini_model(secret_fn)
                text = _gemini_chat(gem_key, model, sysmsg, messages, max_tokens)
                return {"text": text, "provider": "Gemini", "model": model, "local": False}
            except Exception as ex:
                errors.append(f"Gemini: {ex}")

    if "groq" in allow:
        groq_key = secret_fn("groq", "api_key") if secret_fn else None
        if groq_key:
            try:
                model = _groq_model(secret_fn)
                text = _groq_chat(groq_key, model, sysmsg, messages, max_tokens)
                return {"text": text, "provider": "Groq", "model": model, "local": False}
            except Exception as ex:
                errors.append(f"Groq: {ex}")

    raise LLMUnavailable("; ".join(errors) if errors else "No hay IA disponible")


# --------------------------------------------------------------------------- grounding helpers
_MONEY_PAT = re.compile(
    r"\$[\d,]+(?:\.\d+)?"               # $1,234.56
    r"|\b\d[\d,]*(?:\.\d+)?\s*(?:mxn|usd|millones?|mdp|mn\b|k\b)",  # 1,234 MXN / 5 millones
    re.I,
)
_TONNAGE_PAT = re.compile(
    r"\b\d[\d,]*(?:\.\d+)?\s*(?:t\b|ton(?:eladas?|s)?)\b",  # 3.5 t / 120 toneladas
    re.I,
)
_NUMBER_PAT = re.compile(r"\b\d+(?:[,\.]\d+)*\b")


def numbers_ok(ai_text: str, facts_text: str) -> bool:
    """Return True if all numbers in ai_text also appear in facts_text.

    Prevents the LLM from inventing business figures.  A number from ai_text
    is considered 'known' when it appears verbatim in facts_text.
    """
    known = set(_NUMBER_PAT.findall(facts_text))
    for n in _NUMBER_PAT.findall(ai_text):
        if n not in known:
            return False
    return True


def strip_business_numbers(text: str) -> str:
    """Remove sentences that contain monetary amounts or tonnage values.

    Used for open/general answers to prevent the LLM from accidentally
    including business numbers that should only come from the engine.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    clean = [s for s in sentences if not _MONEY_PAT.search(s) and not _TONNAGE_PAT.search(s)]
    return " ".join(clean).strip()
