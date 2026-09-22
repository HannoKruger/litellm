from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import httpx
from pydantic import JsonValue, TypeAdapter, ValidationError

import litellm
from litellm.constants import request_timeout
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.llms.openrouter.common_utils import OpenRouterException
from litellm.secret_managers.main import get_secret_str
from litellm.types.decisions import DecisionQuestion, DecisionsRequest, DecisionsResponse
from litellm.types.utils import LlmProviders
from litellm.utils import client

__all__ = ("adecisions",)

OPENROUTER_DEFAULT_API_BASE: Final = "https://openrouter.ai/api/v1"
PROVIDER_COST_HEADER: Final = "llm_provider-x-litellm-response-cost"
NO_EXTRA_HEADERS: Final[Mapping[str, str]] = MappingProxyType({})
QUESTIONS_ADAPTER: Final = TypeAdapter(Mapping[str, DecisionQuestion])


def openrouter_decisions_url(api_base: str) -> str:
    return f"{api_base.rstrip('/').removesuffix('/v1')}/alpha/decisions"


def _openrouter_api_key(api_key: str | None) -> str | None:
    return (
        api_key
        or litellm.api_key
        or litellm.openrouter_key
        or get_secret_str("OPENROUTER_API_KEY")
        or get_secret_str("OR_API_KEY")
    )


def _parse_decisions_response(content: bytes) -> DecisionsResponse:
    response: Final = DecisionsResponse.model_validate_json(content)
    cost: Final = response.usage.cost if response.usage is not None else None
    if cost is None:
        return response
    cost_headers: Final = {PROVIDER_COST_HEADER: cost}  # mutable-ok: the proxy adds response headers in place
    response._hidden_params["additional_headers"] = cost_headers  # pyright: ignore[reportPrivateUsage]  # cost tracking reads it
    return response


async def _post_openrouter_decisions(
    request: DecisionsRequest,
    api_key: str | None,
    api_base: str,
    timeout: float | httpx.Timeout,
    extra_headers: Mapping[str, str],
) -> DecisionsResponse:
    try:
        http_response: Final = await get_async_httpx_client(llm_provider=LlmProviders.OPENROUTER).post(
            url=openrouter_decisions_url(api_base),
            json=request.model_dump(mode="json", exclude_none=True),
            headers={  # mutable-ok: AsyncHTTPHandler.post only accepts a dict
                **extra_headers,
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )
    except httpx.HTTPStatusError as error:
        raise OpenRouterException(
            status_code=error.response.status_code,
            message=error.response.text,
            headers=error.response.headers,
        ) from error
    return _parse_decisions_response(http_response.content)


def _log_pre_call(logging_obj: object, model: str, provider: str, api_base: str, litellm_call_id: object) -> None:
    if not isinstance(logging_obj, LiteLLMLoggingObj):
        return
    logging_obj.update_from_kwargs(
        kwargs={},  # mutable-ok: legacy logging API takes plain dicts
        model=model,
        optional_params={},  # mutable-ok: legacy logging API takes plain dicts
        litellm_params={  # mutable-ok: update_from_kwargs pops metadata from this dict
            "litellm_call_id": litellm_call_id,
            "api_base": api_base,
        },
        custom_llm_provider=provider,
    )


@client
async def adecisions(
    model: str,
    state: JsonValue,
    questions: Mapping[str, object],
    api_key: str | None = None,
    api_base: str | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    extra_headers: Mapping[str, str] | None = None,
    **kwargs: object,  # kwargs-ok: @client and the router inject logging and routing metadata
) -> DecisionsResponse:
    provider_model, provider, _, _ = litellm.get_llm_provider(model=model, custom_llm_provider=custom_llm_provider)
    if provider != LlmProviders.OPENROUTER.value:
        raise litellm.BadRequestError(
            message=f"Decisions are only supported for openrouter models, got provider={provider}",
            model=model,
            llm_provider=provider,
        )
    try:
        request: Final = DecisionsRequest(
            model=provider_model, state=state, questions=QUESTIONS_ADAPTER.validate_python(questions)
        )
    except ValidationError as error:
        raise litellm.BadRequestError(message=str(error), model=model, llm_provider=provider) from error
    resolved_api_base: Final = api_base or get_secret_str("OPENROUTER_API_BASE") or OPENROUTER_DEFAULT_API_BASE
    _log_pre_call(
        logging_obj=kwargs.get("litellm_logging_obj"),
        model=provider_model,
        provider=provider,
        api_base=openrouter_decisions_url(resolved_api_base),
        litellm_call_id=kwargs.get("litellm_call_id"),
    )
    try:
        return await _post_openrouter_decisions(
            request=request,
            api_key=_openrouter_api_key(api_key),
            api_base=resolved_api_base,
            timeout=timeout or request_timeout,
            extra_headers=extra_headers or NO_EXTRA_HEADERS,
        )
    except Exception as error:
        raise litellm.exception_type(
            model=provider_model,
            custom_llm_provider=provider,
            original_exception=error,
            completion_kwargs={"model": model},  # mutable-ok: exception_type takes plain dicts
            extra_kwargs={},  # mutable-ok: exception_type takes plain dicts
        )
