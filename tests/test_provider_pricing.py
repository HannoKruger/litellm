import os
from typing import Final

import pytest

# Importing provider_pricing starts its refresh thread; keep it from fetching live prices mid-run.
os.environ.setdefault("PROVIDER_PRICING_STARTUP_DELAY", str(24 * 60 * 60))

import litellm  # noqa: E402
import litellm.utils as litellm_utils  # noqa: E402
import provider_pricing  # noqa: E402
from litellm.cost_calculator import rerank_cost  # noqa: E402

RERANKER_CENTS_PER_TOKEN: Final = 2.5e-06

CATALOGUE: Final = (
    {
        "model_name": "example/token-reranker",
        "type": "reranker",
        "pricing": {"type": "input_tokens", "cents_per_input_token": RERANKER_CENTS_PER_TOKEN},
        "max_tokens": 32768,
    },
    {
        "model_name": "example/unpriced-reranker",
        "type": "reranker",
        "pricing": {"type": "time", "cents_per_sec": 0.01, "cents_per_input_token": 1e-06},
        "max_tokens": 32768,
    },
    {
        "model_name": "example/null-priced-reranker",
        "type": "reranker",
        "pricing": {"type": "input_tokens", "cents_per_input_token": None},
        "max_tokens": 32768,
    },
    {
        "model_name": "example/chat-model",
        "type": "text-generation",
        "pricing": {"type": "input_tokens", "cents_per_input_token": 1e-05},
        "max_tokens": 8192,
    },
)


class _FixedCatalogue(provider_pricing.DeepInfraRerankers):
    def fetch(self) -> tuple[dict, ...]:
        return CATALOGUE


@pytest.fixture
def restored_model_cost():
    original_model_cost: Final = dict(litellm.model_cost)
    yield
    litellm.model_cost = original_model_cost
    litellm_utils._invalidate_model_cost_lowercase_map()


def test_only_token_priced_rerankers_are_translated_to_dollars_per_token():
    entries: Final = _FixedCatalogue().entries()

    assert entries == {
        "deepinfra/example/token-reranker": {
            "litellm_provider": "deepinfra",
            "mode": "rerank",
            "input_cost_per_token": pytest.approx(RERANKER_CENTS_PER_TOKEN / 100),
            "output_cost_per_token": 0.0,
            "max_input_tokens": 32768,
        }
    }


def test_registered_reranker_prices_a_deepinfra_rerank_by_input_tokens(restored_model_cost):
    litellm.register_model(_FixedCatalogue().entries(), persist_across_reloads=False)

    prompt_cost, completion_cost = rerank_cost(
        model="example/token-reranker",
        custom_llm_provider="deepinfra",
        billed_units={"total_tokens": 4000},
    )

    assert prompt_cost == pytest.approx(4000 * RERANKER_CENTS_PER_TOKEN / 100)
    assert completion_cost == 0.0
