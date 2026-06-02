# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the chunked execution refactor that underpins D3 staggered training."""

from datetime import datetime

import pytest

from assume.common.forecaster import DemandForecaster
from assume.common.utils import datetime2timestamp
from assume.units.demand import Demand
from tests.utils import index, setup_simple_world


def _seed_world():
    """Return a setup world with one unit operator and one demand unit so
    ``World._validate_setup`` passes (a no-agents world trips an unrelated
    pre-existing bug in that validator).
    """
    world = setup_simple_world()
    world.add_unit_operator("test_operator")
    world.add_unit_instance(
        operator_id="test_operator",
        unit=Demand(
            id="test_unit",
            unit_operator="test_operator",
            min_power=0,
            max_power=-1000,
            technology="demand",
            bidding_strategies={},
            forecaster=DemandForecaster(index, demand=-100),
        ),
    )
    return world


def test_async_run_chunk_round_trips_with_async_run():
    """Splitting one ``async_run`` window into two chunks must end at the same
    clock time as the unsplit reference run (D3 acceptance criterion: chunk
    continuity).
    """
    start = datetime(2022, 1, 1)
    end = datetime(2022, 1, 5)
    midpoint = datetime(2022, 1, 3)
    start_ts = datetime2timestamp(start)
    mid_ts = datetime2timestamp(midpoint)
    end_ts = datetime2timestamp(end)

    # Reference run: single async_run call.
    world_ref = _seed_world()
    with pytest.warns(UserWarning):
        world_ref.loop.run_until_complete(world_ref.async_run(start_ts, end_ts))
    ref_time = world_ref.clock.time

    # Chunked run: two async_run_chunk calls inside one container activation.
    world = _seed_world()

    async def _drive():
        async with world.activate_container():
            await world.async_run_chunk(start_ts, mid_ts)
            mid_clock = world.clock.time
            await world.async_run_chunk(mid_ts, end_ts)
            return mid_clock

    with pytest.warns(UserWarning):
        mid_clock = world.loop.run_until_complete(_drive())

    assert mid_clock >= mid_ts - 1
    assert world.clock.time == ref_time


def test_async_run_chunk_requires_active_container():
    """Calling ``async_run_chunk`` outside ``activate_container`` is a programmer
    error and must raise a clear ``RuntimeError``.
    """
    world = setup_simple_world()
    start_ts = datetime2timestamp(datetime(2022, 1, 1))
    end_ts = datetime2timestamp(datetime(2022, 1, 2))

    async def _call():
        await world.async_run_chunk(start_ts, end_ts)

    with pytest.raises(RuntimeError, match="container"):
        world.loop.run_until_complete(_call())
