from typing import Annotated, Final

import orjson
from fastapi import APIRouter, Depends, Request, Response

from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth, user_api_key_auth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing

router: Final = APIRouter()


@router.post(
    "/v1/decisions",
    dependencies=[Depends(user_api_key_auth)],  # mutable-ok: FastAPI route metadata requires a list
    tags=["decisions"],  # mutable-ok: FastAPI route metadata requires a list
)
@router.post(
    "/decisions",
    dependencies=[Depends(user_api_key_auth)],  # mutable-ok: FastAPI route metadata requires a list
    tags=["decisions"],  # mutable-ok: FastAPI route metadata requires a list
)
async def decisions(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
):
    """
    Structured decisions from a decisions model such as OpenRouter's `typesafe/jev-1.13`.

    Follows OpenRouter's Decisions API: https://openrouter.ai/docs/client-sdks/python/sdks/decisions/README

    ```bash
    curl -X POST "http://localhost:4000/v1/decisions" \\
        -H "Authorization: Bearer sk-1234" \\
        -H "Content-Type: application/json" \\
        -d '{
            "model": "openrouter/typesafe/jev-1.13",
            "state": "Help! My payouts have been failing for 3 days.",
            "questions": {
                "refund": {
                    "type": "noul",
                    "instructions": "Is the customer requesting a refund?",
                    "criteria": {"true": "Asks for money back", "false": "No request for money back"}
                }
            }
        }'
    ```
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    processor: Final = ProxyBaseLLMRequestProcessing(data=orjson.loads(await request.body()))
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="adecisions",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as error:
        raise await processor._handle_llm_api_exception(  # pyright: ignore[reportPrivateUsage]  # every proxy endpoint maps errors through this helper
            e=error,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )
