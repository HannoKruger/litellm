"""Unit tests for litellm.litellm_core_utils.completion_timeout.CompletionTimeout."""

import os
import sys

import httpx
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from litellm.constants import DEEPINFRA_REQUEST_TIMEOUT_SECONDS
from litellm.litellm_core_utils.completion_timeout import CompletionTimeout
from litellm.utils import supports_httpx_timeout


def test_explicit_timeout_wins():
    assert (
        CompletionTimeout.resolve(
            12.5,
            {"timeout": 99.0, "request_timeout": 88.0},
            "openai",
            global_timeout=None,
            supports_httpx_timeout=supports_httpx_timeout,
        )
        == 12.5
    )


def test_kwargs_timeout_when_param_none():
    assert (
        CompletionTimeout.resolve(
            None,
            {"timeout": 21.0},
            "azure_ai",
            global_timeout=None,
            supports_httpx_timeout=supports_httpx_timeout,
        )
        == 21.0
    )


def test_request_timeout_alias_in_kwargs():
    assert (
        CompletionTimeout.resolve(
            None,
            {"request_timeout": 33.0},
            "bedrock",
            global_timeout=None,
            supports_httpx_timeout=supports_httpx_timeout,
        )
        == 33.0
    )


def test_global_timeout_from_litellm_settings():
    assert (
        CompletionTimeout.resolve(
            None,
            {},
            "vertex_ai",
            global_timeout=360.0,
            supports_httpx_timeout=supports_httpx_timeout,
        )
        == 360.0
    )


def test_explicit_global_timeout_6000_is_preserved():
    """The caller passes the explicitly-configured value (or None); an explicit
    6000 must be honored, not silently coerced to 600."""
    assert (
        CompletionTimeout.resolve(
            None,
            {},
            "openai",
            global_timeout=6000.0,
            supports_httpx_timeout=supports_httpx_timeout,
        )
        == 6000.0
    )


def test_explicit_request_timeout_6000_preserved():
    """Explicit deployment/request timeout must not be truncated by the package sentinel."""
    assert (
        CompletionTimeout.resolve(
            None,
            {"request_timeout": 6000.0},
            "openai",
            global_timeout=None,
            supports_httpx_timeout=supports_httpx_timeout,
        )
        == 6000.0
    )


def test_explicit_model_timeout_6000_preserved():
    assert (
        CompletionTimeout.resolve(
            6000.0,
            {"timeout": 1.0, "request_timeout": 2.0},
            "openai",
            global_timeout=None,
            supports_httpx_timeout=supports_httpx_timeout,
        )
        == 6000.0
    )


def test_fallback_600_when_no_global_timeout():
    assert (
        CompletionTimeout.resolve(
            None,
            {},
            "azure_ai",
            global_timeout=None,
            supports_httpx_timeout=supports_httpx_timeout,
        )
        == 600.0
    )


def test_httpx_timeout_coerced_for_provider_without_httpx_timeout_support():
    t = httpx.Timeout(50.0, connect=2.0)
    out = CompletionTimeout.resolve(
        t,
        {},
        "azure_ai",
        global_timeout=None,
        supports_httpx_timeout=supports_httpx_timeout,
    )
    assert out == 50.0
    assert not isinstance(out, httpx.Timeout)


def test_httpx_timeout_preserved_for_openai():
    t = httpx.Timeout(40.0, connect=5.0)
    out = CompletionTimeout.resolve(
        t,
        {},
        "openai",
        global_timeout=None,
        supports_httpx_timeout=supports_httpx_timeout,
    )
    assert out is t
    assert isinstance(out, httpx.Timeout)


@pytest.mark.parametrize(
    ("model_timeout", "kwargs", "global_timeout", "expected"),
    [
        (600.0, {}, None, DEEPINFRA_REQUEST_TIMEOUT_SECONDS),
        (None, {}, 6000.0, DEEPINFRA_REQUEST_TIMEOUT_SECONDS),
        (None, {}, None, DEEPINFRA_REQUEST_TIMEOUT_SECONDS),
        (httpx.Timeout(900.0, connect=5.0), {}, None, DEEPINFRA_REQUEST_TIMEOUT_SECONDS),
        (None, {"request_timeout": 45.0}, None, 45.0),
        (60.0, {}, None, 60.0),
    ],
)
def test_deepinfra_calls_never_wait_longer_than_the_deepinfra_timeout(model_timeout, kwargs, global_timeout, expected):
    resolved = CompletionTimeout.resolve(
        model_timeout,
        kwargs,
        "deepinfra",
        global_timeout=global_timeout,
        supports_httpx_timeout=supports_httpx_timeout,
    )

    assert resolved == expected


def test_deepinfra_timeout_does_not_shorten_other_providers():
    resolved = CompletionTimeout.resolve(
        600.0,
        {},
        "openai",
        global_timeout=None,
        supports_httpx_timeout=supports_httpx_timeout,
    )

    assert resolved == 600.0
