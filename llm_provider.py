"""
Provider configuration for LLM-backed classification.

All supported providers speak the OpenAI-compatible /chat/completions
protocol, so a single client class drives them all. Selection is done via
env vars -- no code changes needed to swap providers:

    LLM_PROVIDER=groq         LLM_API_KEY=gsk_...
    LLM_PROVIDER=openrouter   LLM_API_KEY=sk-or-...
    LLM_PROVIDER=gemini       LLM_API_KEY=<gemini key>
    LLM_PROVIDER=ollama       (no key needed; local server)

GEMINI_API_KEY is still honored for backward compatibility.
"""

from __future__ import annotations

import os

# base_url, default model, needs_api_key
PROVIDERS: dict[str, tuple[str, str, bool]] = {
    "groq":       ("https://api.groq.com/openai/v1",
                   "llama-3.3-70b-versatile", True),
    "openrouter": ("https://openrouter.ai/api/v1",
                   "meta-llama/llama-3.3-70b-instruct:free", True),
    # Gemini's official OpenAI-compatible endpoint -- kept so existing
    # GEMINI_API_KEY setups work through the same code path.
    "gemini":     ("https://generativelanguage.googleapis.com/v1beta/openai/",
                   "gemini-2.0-flash", True),
    "ollama":     ("http://localhost:11434/v1",
                   "llama3.1", False),
}


class ProviderConfig:
    def __init__(self, provider: str | None = None,
                 api_key: str | None = None,
                 model: str | None = None,
                 base_url: str | None = None):
        self.provider = (provider or os.environ.get("LLM_PROVIDER")
                         or ("gemini" if os.environ.get("GEMINI_API_KEY") else "groq")
                         ).lower().strip()
        if self.provider not in PROVIDERS:
            raise ValueError(
                f"Unknown LLM_PROVIDER '{self.provider}'. "
                f"Valid options: {', '.join(sorted(PROVIDERS))}"
            )

        url, default_model, needs_key = PROVIDERS[self.provider]
        self.base_url = base_url or os.environ.get("LLM_BASE_URL") or url
        self.model = model or os.environ.get("LLM_MODEL") or default_model

        self.api_key = (
            api_key
            or os.environ.get("LLM_API_KEY")
            or os.environ.get("GEMINI_API_KEY")  # backward compat
        )
        if needs_key and not self.api_key:
            raise RuntimeError(
                f"No API key for provider '{self.provider}'. "
                "Set LLM_API_KEY (or GEMINI_API_KEY for gemini)."
            )
        if not needs_key:
            self.api_key = self.api_key or "not-needed"
