"""Models flagged `model_info.exclude_from_budgets: true` log their full cost but never touch a budget.

A flat-rate subscription served through the proxy still has a real per-token price worth
recording, yet its usage must neither be capped nor eat into the budget that caps paid models.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from litellm.router import Router

EXCLUDE_FROM_BUDGETS_FLAG: Final = "exclude_from_budgets"


def _field(mapping: object, key: str) -> object:
    if not isinstance(mapping, Mapping):
        return None
    return mapping.get(key)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]  # model_info and litellm_params are untyped dicts; the one value read is checked by its caller


def is_excluded_from_budgets(model_info: object) -> bool:
    return _field(model_info, EXCLUDE_FROM_BUDGETS_FLAG) is True


def model_group_is_excluded_from_budgets(model_name: str, llm_router: "Router") -> bool:
    deployments: Final = llm_router.get_model_list(model_name=model_name) or []
    return len(deployments) > 0 and all(
        is_excluded_from_budgets(_field(deployment, "model_info")) for deployment in deployments
    )


def served_deployment_is_excluded_from_budgets(call_kwargs: object) -> bool:
    return is_excluded_from_budgets(_field(_field(call_kwargs, "litellm_params"), "model_info"))
