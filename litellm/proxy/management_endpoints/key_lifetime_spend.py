"""Lifetime spend per virtual key, read from the daily spend aggregates.

A key's own `spend` column only covers the current budget window: `reset_budget_job`
zeroes it every time `budget_reset_at` passes, and `/key/{key}/reset_spend` zeroes it
on demand. `LiteLLM_DailyUserSpend` keeps one row per key, day, model and endpoint and
is never reset, so summing it gives the total a key has ever spent. It is the same
table the usage pages read, so the two agree.
"""

from collections.abc import Mapping, Sequence
from typing import Final, Protocol

from litellm.proxy.utils import PrismaClient


class _DailyUserSpendTable(Protocol):
    """The subset of the Prisma daily-user-spend table API this module uses."""

    async def group_by(
        self,
        *,
        by: Sequence[str],
        where: Mapping[str, object],
        sum: Mapping[str, bool],
    ) -> Sequence[Mapping[str, object]]: ...


def _daily_user_spend_table(prisma_client: PrismaClient) -> _DailyUserSpendTable:
    return prisma_client.db.litellm_dailyuserspend


def _row_spend(row: Mapping[str, object]) -> float:
    """Read the summed spend out of a Prisma `group_by` row, which nests it under `_sum`."""
    totals: Final = row.get("_sum")
    raw: Final = totals.get("spend") if isinstance(totals, Mapping) else row.get("spend")
    return float(raw) if isinstance(raw, (int, float)) else 0.0


async def fetch_lifetime_spend_by_key(
    prisma_client: PrismaClient,
    tokens: Sequence[str],
) -> Mapping[str, float]:
    """Total spend ever recorded per key hash, for the given hashes.

    Keys with no recorded usage are absent from the result rather than mapped to 0.0,
    so a caller can tell "never used" apart from "used, rounded to zero".
    """
    if not tokens:
        return {}

    rows: Final = await _daily_user_spend_table(prisma_client).group_by(
        by=["api_key"],
        where={"api_key": {"in": list(tokens)}},
        sum={"spend": True},
    )
    return {
        str(api_key): _row_spend(row)
        for row in rows
        if isinstance(row, Mapping) and (api_key := row.get("api_key")) is not None
    }


async def fetch_lifetime_spend_for_key(prisma_client: PrismaClient, token: str) -> float:
    """Total spend ever recorded for a single key hash. Zero when the key never spent."""
    totals: Final = await fetch_lifetime_spend_by_key(prisma_client, [token])
    return totals.get(token, 0.0)
