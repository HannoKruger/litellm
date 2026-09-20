from collections.abc import Mapping, Sequence
from types import SimpleNamespace

import pytest

from litellm.proxy.management_endpoints.key_lifetime_spend import (
    fetch_lifetime_spend_by_key,
    fetch_lifetime_spend_for_key,
)


class FakeDailyUserSpendTable:
    """Records the query it was asked for and replays canned `group_by` rows."""

    def __init__(self, rows: Sequence[Mapping[str, object]]):
        self.rows = rows
        self.calls: list[Mapping[str, object]] = []

    async def group_by(
        self,
        *,
        by: Sequence[str],
        where: Mapping[str, object],
        sum: Mapping[str, bool],
    ) -> Sequence[Mapping[str, object]]:
        self.calls.append({"by": list(by), "where": where, "sum": sum})
        return self.rows


def fake_prisma(rows: Sequence[Mapping[str, object]]) -> tuple[object, FakeDailyUserSpendTable]:
    table = FakeDailyUserSpendTable(rows)
    return SimpleNamespace(db=SimpleNamespace(litellm_dailyuserspend=table)), table


@pytest.mark.asyncio
async def test_sums_every_day_and_model_row_into_one_total_per_key():
    prisma_client, table = fake_prisma(
        [
            {"api_key": "hash-a", "_sum": {"spend": 12.5}},
            {"api_key": "hash-b", "_sum": {"spend": 0.25}},
        ]
    )

    totals = await fetch_lifetime_spend_by_key(prisma_client, ["hash-a", "hash-b"])

    assert totals == {"hash-a": 12.5, "hash-b": 0.25}
    assert table.calls == [
        {
            "by": ["api_key"],
            "where": {"api_key": {"in": ["hash-a", "hash-b"]}},
            "sum": {"spend": True},
        }
    ]


@pytest.mark.asyncio
async def test_omits_a_key_the_aggregates_never_recorded_so_unused_is_not_reported_as_zero():
    prisma_client, _ = fake_prisma([{"api_key": "hash-a", "_sum": {"spend": 1.0}}])

    totals = await fetch_lifetime_spend_by_key(prisma_client, ["hash-a", "hash-unused"])

    assert "hash-unused" not in totals


@pytest.mark.asyncio
async def test_reads_a_flat_spend_row_when_the_client_does_not_nest_under_sum():
    prisma_client, _ = fake_prisma([{"api_key": "hash-a", "spend": 3.75}])

    assert await fetch_lifetime_spend_by_key(prisma_client, ["hash-a"]) == {"hash-a": 3.75}


@pytest.mark.asyncio
async def test_queries_nothing_when_asked_for_no_keys():
    prisma_client, table = fake_prisma([{"api_key": "hash-a", "_sum": {"spend": 1.0}}])

    assert await fetch_lifetime_spend_by_key(prisma_client, []) == {}
    assert table.calls == []


@pytest.mark.asyncio
async def test_single_key_lookup_returns_zero_for_a_key_that_never_spent():
    prisma_client, _ = fake_prisma([])

    assert await fetch_lifetime_spend_for_key(prisma_client, "hash-unused") == 0.0


@pytest.mark.asyncio
async def test_single_key_lookup_returns_that_key_total():
    prisma_client, _ = fake_prisma(
        [
            {"api_key": "hash-a", "_sum": {"spend": 8.2179}},
            {"api_key": "hash-b", "_sum": {"spend": 99.0}},
        ]
    )

    assert await fetch_lifetime_spend_for_key(prisma_client, "hash-a") == 8.2179
