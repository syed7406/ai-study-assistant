from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import List

import httpx

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover - optional dependency
    genai = None


@dataclass(frozen=True)
class LLMConfig:
    provider: str = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1:latest")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "").strip()


def _build_prompt(question: str, context: str) -> str:
    return (
        "You are a study assistant. Answer the question based on the context below. "
        "Use the information from the context to give the best answer you can. "
        "If the context is not relevant to the question, say \"I don't know based on your documents.\"\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )


def _extractive_fallback(question: str, context_chunks: List[str]) -> str:
    context = "\n\n".join(context_chunks).strip()
    if not context:
        return "I don't know based on your documents."

    question_terms = {
        term
        for term in re.findall(r"[a-zA-Z0-9_]+", question.lower())
        if len(term) > 2
    }
    if not question_terms:
        return context_chunks[0][:500]

    scored_sentences: list[tuple[int, int, str]] = []
    for chunk in context_chunks:
        for sentence_index, sentence in enumerate(re.split(r"(?<=[.!?])\s+", chunk)):
            words = set(re.findall(r"[a-zA-Z0-9_]+", sentence.lower()))
            score = len(question_terms & words)
            if score:
                scored_sentences.append((score, sentence_index, sentence.strip()))

    if not scored_sentences:
        return context_chunks[0][:500]

    scored_sentences.sort(key=lambda item: (-item[0], item[1]))
    top_sentences = [sentence for _, _, sentence in scored_sentences[:3]]
    return " ".join(top_sentences).strip() or "I don't know based on your documents."


def _call_ollama(prompt: str) -> str:
    config = LLMConfig()
    response = httpx.post(
        f"{config.ollama_base_url}/api/generate",
        json={
            "model": config.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2},
        },
        timeout=60.0,
    )
    response.raise_for_status()
    data = response.json()
    return (data.get("response") or "").strip()


def check_ollama_connection() -> dict:
    config = LLMConfig()
    try:
        response = httpx.get(f"{config.ollama_base_url}/api/tags", timeout=5.0)
        response.raise_for_status()
        data = response.json()
        models = data.get("models", [])
        return {
            "connected": True,
            "provider": "ollama",
            "base_url": config.ollama_base_url,
            "model": config.ollama_model,
            "available_models": [item.get("name", "") for item in models if isinstance(item, dict)],
        }
    except Exception as exc:
        return {
            "connected": False,
            "provider": "ollama",
            "base_url": config.ollama_base_url,
            "model": config.ollama_model,
            "error": str(exc),
        }


def _call_gemini(prompt: str) -> str:
    config = LLMConfig()
    if not config.gemini_api_key or genai is None:
        return ""

    genai.configure(api_key=config.gemini_api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0.2, "max_output_tokens": 512},
    )
    return (getattr(response, "text", "") or "").strip()


def build_answer(question: str, context_chunks: List[str]) -> str:
    config = LLMConfig()
    context = "\n\n".join(context_chunks).strip()
    if not context:
        return "I don't know based on your documents."

    prompt = _build_prompt(question, context)

    if config.provider == "gemini":
        text = _call_gemini(prompt)
        return text or _extractive_fallback(question, context_chunks)

    if config.provider == "ollama":
        try:
            text = _call_ollama(prompt)
            return text or _extractive_fallback(question, context_chunks)
        except Exception:
            return _extractive_fallback(question, context_chunks)

    return _extractive_fallback(question, context_chunks)
