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

Model catalogs on free/hosted tiers (Groq, OpenRouter, NVIDIA NIM) churn
often -- models get deprecated, delisted, or moved to download-only without
much warning. `ProviderConfig.verify_model_available()` checks the
configured model against the provider's live catalog at startup, so a bad
model name fails loudly and clearly instead of surfacing as a 404/410 deep
inside a classification request.
"""

from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)

# base_url, default model, needs_api_key
PROVIDERS: dict[str, tuple[str, str, bool]] = {
    "groq":       ("https://api.groq.com/openai/v1",
                   "openai/gpt-oss-120b", True),
    # NOTE: OpenRouter's free-tier catalog churns especially fast -- models
    # get delisted/repriced with no notice. Don't trust this default blindly;
    # verify_model_available() below will catch it at startup if it's stale.
    "openrouter": ("https://openrouter.ai/api/v1",
                   "meta-llama/llama-3.1-8b-instant:free", True),
    # Gemini's official OpenAI-compatible endpoint -- kept so existing
    # GEMINI_API_KEY setups work through the same code path.
    "gemini":     ("https://generativelanguage.googleapis.com/v1beta/openai/",
                   "gemini-2.0-flash", True),
    "ollama":     ("http://localhost:11434/v1",
                   "llama3.1", False),
}

# Providers whose /models catalog is known to genuinely go missing/timeout
# often enough in normal operation (local server not running, etc.) that a
# failed *connection* shouldn't be treated as fatal -- only a confirmed
# "model not in catalog" response should be.
_MODEL_CHECK_TIMEOUT_SECONDS = float(os.environ.get("LLM_MODEL_CHECK_TIMEOUT", "5"))


class ProviderConfig:

    def __init__(self, provider: str | None=None,
                 api_key: str | None=None,
                 model: str | None=None,
                 base_url: str | None=None):
        self.provider = (provider or os.environ.get("LLM_PROVIDER")
                         or "groq"
                         ).lower().strip()
        if self.provider not in PROVIDERS:
            raise ValueError(
                f"Unknown LLM_PROVIDER '{self.provider}'. "
                f"Valid options: {', '.join(sorted(PROVIDERS))}"
            )

        url, default_model, needs_key = PROVIDERS[self.provider]
        self.base_url = base_url or os.environ.get("LLM_BASE_URL") or url
        self.model = model or os.environ.get("LLM_MODEL") or default_model
        self.needs_key = needs_key

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

    def verify_model_available(self) -> None:
        """
        Confirm `self.model` actually exists in the provider's live catalog.

        Raises RuntimeError if the provider was reachable and returned a
        model list that does NOT contain our configured model -- this is a
        real misconfiguration and should stop startup (or trigger the
        MockClassifier fallback in main.py) rather than fail later on the
        first real classification request.

        Does NOT raise if the catalog endpoint itself is unreachable
        (network error, timeout, 5xx) -- that's treated as inconclusive and
        only logged, since it may just be transient and blocking startup on
        it would be worse than proceeding optimistically.
        """
        if os.environ.get("LLM_SKIP_MODEL_CHECK", "").lower() in ("1", "true", "yes"):
            logger.info("Skipping model availability check (LLM_SKIP_MODEL_CHECK set).")
            return

        models_url = f"{self.base_url.rstrip('/')}/models"
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.needs_key else {}

        try:
            resp = httpx.get(models_url, headers=headers, timeout=_MODEL_CHECK_TIMEOUT_SECONDS)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:  # noqa: BLE001 -- deliberately broad, see docstring
            logger.warning(
                "Could not verify model catalog for provider '%s' at %s (%s). "
                "Proceeding without verification -- if the model is actually "
                "invalid, this will surface as an error on the first "
                "classification request instead.",
                self.provider, models_url, exc,
            )
            return

        available = {m["id"] for m in payload.get("data", []) if "id" in m}
        if not available:
            logger.warning(
                "Provider '%s' returned an empty/unrecognized model list from %s; "
                "skipping verification.", self.provider, models_url,
            )
            return

        if self.model not in available:
            sample = sorted(available)[:20]
            raise RuntimeError(
                f"Model '{self.model}' is not available on provider "
                f"'{self.provider}' ({models_url}). "
                f"Sample of currently available models: {sample}"
            )

        logger.info("Verified model '%s' is available on provider '%s'.",
                    self.model, self.provider)