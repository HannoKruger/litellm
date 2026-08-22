"""Keep LiteLLM's cost map correct for providers it prices badly or not at all.

LiteLLM prices models from one static JSON catalogue (upstream's, refreshed
periodically). Providers that ship models faster than that catalogue is updated
end up either missing -- billed at $0 -- or carrying stale prices. As of
2026-08-19 DeepInfra was missing 91 models and mispriced 29 of the ones it had,
in both directions (MythoMax-L2-13b under-reported 5x, gpt-oss-120b
over-reported).

Providers that publish their own live prices are the fix: fetch the catalogue,
translate it, and hand it to ``litellm.register_model``. Registrations made that
way carry ``persist_across_reloads=True``, so they are replayed every time the
cost map refreshes rather than being clobbered by the next upstream fetch.

Wiring (LiteLLM_Config.litellm_settings in the proxy DB, set via the Admin UI
or POST /config/update -- not config.yaml)::

    litellm_settings:
      callbacks: ["smtp_email", "provider_pricing.sync"]

The callbacks list is resolved through ``get_instance_fn`` at config load, so
importing this module is what schedules the first sync. ``sync`` is a
``CustomLogger`` purely to satisfy that contract -- it logs nothing.

Adding a provider is one ``PricingSource`` subclass plus an entry in ``SOURCES``.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.request
from typing import Any, Final

import litellm
from litellm.integrations.custom_logger import CustomLogger

try:
    from litellm._logging import verbose_logger
except ImportError:  # pragma: no cover - logging shape differs across versions
    import logging

    verbose_logger = logging.getLogger("litellm")

REFRESH_SECONDS: Final = int(os.getenv("PROVIDER_PRICING_REFRESH_SECONDS", str(6 * 60 * 60)))
FETCH_TIMEOUT: Final = int(os.getenv("PROVIDER_PRICING_TIMEOUT", "30"))
STARTUP_DELAY_SECONDS: Final = int(os.getenv("PROVIDER_PRICING_STARTUP_DELAY", "45"))
CACHE_PREFIX: Final = "provider_pricing:last_known:"
PER_MILLION: Final = 1_000_000


# --------------------------------------------------------------------------
# Last-known-good cache
# --------------------------------------------------------------------------
# A restart during a provider outage must not silently fall back to upstream's
# wrong prices. Redis is already a dependency of this proxy and outlives
# container recreation, so the last good catalogue is cached there. Failure to
# reach Redis is never fatal -- it only costs us the fallback.
class _Cache:
    def __init__(self) -> None:
        self._client: Any = None
        self._tried = False

    def _redis(self) -> Any:
        if self._tried:
            return self._client
        self._tried = True
        host = os.getenv("REDIS_HOST")
        if not host:
            return None
        try:
            import redis

            self._client = redis.Redis(
                host=host,
                port=int(os.getenv("REDIS_PORT", "6379")),
                password=os.getenv("REDIS_PASSWORD") or None,
                socket_timeout=5,
                decode_responses=True,
            )
            self._client.ping()
        except Exception as exc:  # noqa: BLE001 - cache is best-effort
            verbose_logger.warning("provider_pricing: cache unavailable (%s)", exc)
            self._client = None
        return self._client

    def get(self, name: str) -> dict | None:
        client = self._redis()
        if client is None:
            return None
        try:
            raw = client.get(CACHE_PREFIX + name)
            return json.loads(raw) if raw else None
        except Exception:  # noqa: BLE001
            return None

    def put(self, name: str, entries: dict) -> None:
        client = self._redis()
        if client is None:
            return
        try:
            client.set(CACHE_PREFIX + name, json.dumps(entries))
        except Exception:  # noqa: BLE001
            pass


_cache: Final = _Cache()


# --------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------
class PricingSource:
    """One provider that publishes machine-readable prices."""

    name: str
    catalogue: str

    def fetch(self) -> Any:
        req = urllib.request.Request(self.catalogue, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT) as response:
            return json.load(response)

    def entries(self) -> dict[str, dict]:
        raise NotImplementedError


class DeepInfra(PricingSource):
    """DeepInfra publishes prices unauthenticated, in USD per 1M tokens."""

    name = "deepinfra"
    catalogue = "https://api.deepinfra.com/v1/openai/models"

    # Only token-priced modalities are translatable. image-gen / video-gen /
    # tts / stt bill on other axes entirely, so they are skipped rather than
    # guessed at -- a wrong price is worse than a missing one.
    MODES: Final = {"chat": "chat", "embed": "embedding"}

    def entries(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for model in self.fetch().get("data", []):
            meta = model.get("metadata") or {}
            pricing = meta.get("pricing") or {}
            if "input_tokens" not in pricing:
                continue
            tags = meta.get("tags") or []
            mode = self.MODES.get(next((t for t in tags if t in self.MODES), ""))
            if mode is None:
                continue

            entry: dict[str, Any] = {
                "litellm_provider": self.name,
                "mode": mode,
                "input_cost_per_token": pricing["input_tokens"] / PER_MILLION,
                "output_cost_per_token": pricing.get("output_tokens", 0) / PER_MILLION,
            }
            if pricing.get("cache_read_tokens") is not None:
                entry["cache_read_input_token_cost"] = pricing["cache_read_tokens"] / PER_MILLION
            if meta.get("context_length"):
                entry["max_input_tokens"] = meta["context_length"]
            if meta.get("max_tokens"):
                entry["max_output_tokens"] = meta["max_tokens"]
            if mode == "chat":
                entry["supports_prompt_caching"] = "prompt_cache" in tags
                entry["supports_reasoning"] = "reasoning" in tags
                entry["supports_vision"] = "vision" in tags
                entry["supports_function_calling"] = True

            out[f"{self.name}/{model['id']}"] = entry
        return out


SOURCES: Final[tuple[PricingSource, ...]] = (DeepInfra(),)


# --------------------------------------------------------------------------
# Sync
# --------------------------------------------------------------------------
def sync_source(source: PricingSource) -> int:
    """Register one provider's live prices. Returns how many models were priced.

    Fail-open: a provider that cannot be reached falls back to its last known
    catalogue, and failing that is skipped. Pricing never blocks the proxy.
    """
    try:
        entries = source.entries()
        if not entries:
            raise ValueError("provider returned no priced models")
        _cache.put(source.name, entries)
    except Exception as exc:  # noqa: BLE001 - never fatal
        entries = _cache.get(source.name) or {}
        if entries:
            verbose_logger.warning(
                "provider_pricing: %s unreachable (%s); using %d cached prices",
                source.name,
                exc,
                len(entries),
            )
        else:
            verbose_logger.warning(
                "provider_pricing: %s unreachable (%s) and no cache; "
                "upstream prices left in place",
                source.name,
                exc,
            )
            return 0

    litellm.register_model(entries)
    return len(entries)


def sync_all() -> None:
    for source in SOURCES:
        count = sync_source(source)
        if count:
            verbose_logger.info("provider_pricing: registered %d %s models", count, source.name)


def _refresh_loop(stop: threading.Event) -> None:
    # The first sync is deliberately inside this thread, not at import. Import
    # happens during proxy config load, and registering a catalogue there races
    # the proxy's own cost-map load: the port never binds and nothing is logged.
    # Deferring past STARTUP_DELAY_SECONDS keeps pricing entirely off the boot
    # path, which also makes a provider outage structurally unable to delay it.
    stop.wait(STARTUP_DELAY_SECONDS)
    while True:
        try:
            sync_all()
        except Exception as exc:  # noqa: BLE001 - a dead thread means silent drift
            verbose_logger.warning("provider_pricing: sync failed (%s)", exc)
        if stop.wait(REFRESH_SECONDS):
            return


class ProviderPricingSync(CustomLogger):
    """Schedules the price sync. Implements no logging hooks by design.

    Construction only starts a daemon thread -- it performs no network or
    registration work, so importing this module cannot delay or break proxy
    startup no matter how the provider behaves.
    """

    _started = False
    _lock: Final = threading.Lock()

    def __init__(self) -> None:
        super().__init__()
        self.start()

    @classmethod
    def start(cls) -> None:
        # config load can construct callbacks more than once; one thread only.
        with cls._lock:
            if cls._started:
                return
            cls._started = True
        stop = threading.Event()
        threading.Thread(
            target=_refresh_loop,
            args=(stop,),
            name="provider-pricing-refresh",
            daemon=True,
        ).start()


sync: Final = ProviderPricingSync()
