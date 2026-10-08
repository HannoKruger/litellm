import pytest

from litellm import Router
from litellm.proxy.auth.auth_checks import _is_model_cost_zero
from litellm.proxy.spend_tracking.budget_exclusion import (
    is_excluded_from_budgets,
    model_group_is_excluded_from_budgets,
    served_deployment_is_excluded_from_budgets,
)


def _router(model_info: dict) -> Router:
    return Router(
        model_list=[
            {
                "model_name": "proxy/anthropic/*",
                "litellm_params": {"model": "anthropic/*", "api_base": "http://proxy", "api_key": "test"},
                "model_info": {"id": "proxy-wildcard", **model_info},
            },
            {
                "model_name": "paid-model",
                "litellm_params": {"model": "anthropic/claude-opus-5-5", "api_key": "test"},
                "model_info": {"id": "paid"},
            },
        ]
    )


@pytest.mark.parametrize(
    ("model_info", "expected"),
    [({"exclude_from_budgets": True}, True), ({"exclude_from_budgets": "true"}, False), ({}, False), (None, False)],
)
def test_is_excluded_from_budgets_needs_a_literal_true(model_info, expected):
    assert is_excluded_from_budgets(model_info) is expected


def test_wildcard_model_flagged_as_excluded_resolves_for_a_concrete_name():
    router = _router({"exclude_from_budgets": True})

    assert model_group_is_excluded_from_budgets("proxy/anthropic/claude-haiku-5-5", router) is True
    assert model_group_is_excluded_from_budgets("paid-model", router) is False
    assert model_group_is_excluded_from_budgets("unknown-model", router) is False


def test_budget_checks_skip_an_excluded_model_but_not_a_priced_one():
    router = _router({"exclude_from_budgets": True})

    assert _is_model_cost_zero(model="proxy/anthropic/claude-haiku-5-5", llm_router=router) is True
    assert _is_model_cost_zero(model="paid-model", llm_router=router) is False
    assert _is_model_cost_zero(model=["proxy/anthropic/claude-haiku-5-5", "paid-model"], llm_router=router) is False


def test_budget_checks_apply_to_the_wildcard_without_the_flag():
    assert _is_model_cost_zero(model="proxy/anthropic/claude-haiku-5-5", llm_router=_router({})) is False


def test_served_deployment_reads_model_info_from_call_kwargs():
    assert served_deployment_is_excluded_from_budgets(
        {"litellm_params": {"model_info": {"exclude_from_budgets": True}}}
    )
    assert not served_deployment_is_excluded_from_budgets({"litellm_params": None})
    assert not served_deployment_is_excluded_from_budgets({})
