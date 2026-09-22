import asyncio
import json
from collections.abc import Mapping
from datetime import datetime
from typing import Final, cast

import httpx
import pytest
import respx

import litellm
from litellm import Router
from litellm.decisions.main import openrouter_decisions_url
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.decisions import ChoiceAnswer, DecisionsResponse, NoulAnswer
from litellm.types.utils import StandardLoggingPayload

DECISIONS_URL: Final = "https://openrouter.ai/api/alpha/decisions"
QUESTIONS: Final = {
    "refund": {
        "type": "noul",
        "instructions": "Is the customer requesting a refund?",
        "criteria": {"true": "Asks for money back", "false": "No request for money back"},
    },
    "team": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {"billing": "Payments", "technical": "Bugs"},
    },
}
UPSTREAM_USAGE: Final = {"input_tokens": 399, "output_tokens": 61, "cost": 0.000016758}
UPSTREAM_RESPONSE: Final = {
    "id": "gen-dec-1",
    "model": "typesafe/jev-1.13-20260917",
    "provider": "TypeSafe",
    "answers": {
        "refund": {"type": "noul", "noul": 0.97},
        "team": {
            "type": "choice",
            "choice": "billing",
            "probabilities": {"billing": 0.98, "technical": 0.02},
            "confidence": 0.97,
        },
    },
    "usage": UPSTREAM_USAGE,
}


@pytest.fixture(autouse=True)
def httpx_transport_for_respx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()


class _CapturingLogger(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.payloads: list[StandardLoggingPayload] = []  # mutable-ok: collects logged payloads across callbacks

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.payloads.append(
            cast(StandardLoggingPayload, kwargs["standard_logging_object"])
        )  # cast-ok: litellm always sets it


@pytest.mark.parametrize(
    ("api_base", "expected"),
    [
        ("https://openrouter.ai/api/v1", DECISIONS_URL),
        ("https://openrouter.ai/api/v1/", DECISIONS_URL),
        ("https://gateway.example/openrouter/api", "https://gateway.example/openrouter/api/alpha/decisions"),
    ],
)
def test_openrouter_decisions_url_replaces_the_chat_version_segment(api_base: str, expected: str):
    assert openrouter_decisions_url(api_base) == expected


@pytest.mark.asyncio
async def test_adecisions_posts_the_decisions_payload_and_returns_typed_answers():
    with respx.mock(assert_all_called=True) as respx_mock:
        route: Final = respx_mock.post(DECISIONS_URL).mock(return_value=httpx.Response(200, json=UPSTREAM_RESPONSE))
        response: Final = await litellm.adecisions(
            model="openrouter/typesafe/jev-1.13",
            state="My payouts failed, I want my money back.",
            questions=QUESTIONS,
            api_key="sk-or-test",
        )

    sent: Final = route.calls.last.request
    assert sent.headers["Authorization"] == "Bearer sk-or-test"
    assert json.loads(sent.content) == {
        "model": "typesafe/jev-1.13",
        "state": "My payouts failed, I want my money back.",
        "questions": QUESTIONS,
    }
    assert response.answers["refund"] == NoulAnswer(type="noul", noul=0.97)
    assert isinstance(response.answers["team"], ChoiceAnswer)
    assert response.answers["team"].choice == "billing"


@pytest.mark.asyncio
async def test_adecisions_logs_openrouter_reported_cost_and_tokens(monkeypatch: pytest.MonkeyPatch):
    logger: Final = _CapturingLogger()
    monkeypatch.setattr(litellm, "callbacks", [logger])
    with respx.mock(assert_all_called=True) as respx_mock:
        respx_mock.post(DECISIONS_URL).mock(return_value=httpx.Response(200, json=UPSTREAM_RESPONSE))
        await litellm.adecisions(
            model="openrouter/typesafe/jev-1.13", state="x", questions=QUESTIONS, api_key="sk-or-test"
        )
    for _ in range(50):
        if logger.payloads:
            break
        await asyncio.sleep(0.05)

    payload: Final = logger.payloads[-1]
    assert payload["response_cost"] == pytest.approx(UPSTREAM_USAGE["cost"])
    assert payload["prompt_tokens"] == UPSTREAM_USAGE["input_tokens"]
    assert payload["completion_tokens"] == UPSTREAM_USAGE["output_tokens"]
    assert payload["call_type"] == "adecisions"


@pytest.mark.asyncio
async def test_adecisions_rejects_non_openrouter_models_without_calling_upstream():
    with respx.mock(assert_all_called=False) as respx_mock:
        route: Final = respx_mock.post(DECISIONS_URL)
        with pytest.raises(litellm.BadRequestError, match="only supported for openrouter"):
            await litellm.adecisions(model="deepinfra/openai/gpt-oss-20b", state="x", questions=QUESTIONS)
    assert not route.called


@pytest.mark.asyncio
async def test_adecisions_rejects_unknown_question_types_as_bad_request():
    with pytest.raises(litellm.BadRequestError):
        await litellm.adecisions(
            model="openrouter/typesafe/jev-1.13",
            state="x",
            questions={"q": {"type": "essay", "instructions": "Write one", "criteria": {}}},
            api_key="sk-or-test",
        )


@pytest.mark.asyncio
async def test_adecisions_maps_upstream_auth_failure_to_authentication_error():
    with respx.mock(assert_all_called=True) as respx_mock:
        respx_mock.post(DECISIONS_URL).mock(
            return_value=httpx.Response(401, json={"error": {"message": "User not found.", "code": 401}})
        )
        with pytest.raises(litellm.AuthenticationError, match="User not found"):
            await litellm.adecisions(
                model="openrouter/typesafe/jev-1.13", state="x", questions=QUESTIONS, api_key="sk-or-bad"
            )


@pytest.mark.asyncio
async def test_router_adecisions_uses_the_deployment_credentials():
    router: Final = Router(
        model_list=[
            {
                "model_name": "jev",
                "litellm_params": {"model": "openrouter/typesafe/jev-1.13", "api_key": "sk-or-deployment"},
            }
        ]
    )
    with respx.mock(assert_all_called=True) as respx_mock:
        route: Final = respx_mock.post(DECISIONS_URL).mock(return_value=httpx.Response(200, json=UPSTREAM_RESPONSE))
        response: Final = await router.adecisions(model="jev", state="x", questions=QUESTIONS)

    assert route.calls.last.request.headers["Authorization"] == "Bearer sk-or-deployment"
    assert json.loads(route.calls.last.request.content)["model"] == "typesafe/jev-1.13"
    assert isinstance(response, DecisionsResponse)
    assert set(response.answers) == set(QUESTIONS)
