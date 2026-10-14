# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime

import pandas as pd
import pytest
from dateutil import rrule as rr
from dateutil.relativedelta import relativedelta as rd
from mango import RoleAgent, activate, create_tcp_container, sender_addr
from mango.util.clock import ExternalClock
from mango.util.termination_detection import tasks_complete_or_sleeping

from assume.common.market_objects import MarketConfig
from assume.common.utils import datetime2timestamp
from assume.markets.base_market import MarketProduct, MarketRole

start = datetime(2020, 1, 1)
end = datetime(2020, 12, 2)


def test_market_config_hash_consistent():
    """Hash of the same MarketConfig instance is stable across calls."""
    mc = MarketConfig(market_id="EOM")
    assert hash(mc) == hash(mc)


def test_market_config_hash_equal_instances():
    """Two MarketConfig instances with identical values hash the same."""
    mc1 = MarketConfig(market_id="EOM", minimum_bid_price=-100.0)
    mc2 = MarketConfig(market_id="EOM", minimum_bid_price=-100.0)
    assert hash(mc1) == hash(mc2)


def test_market_config_hash_different_instances():
    """MarketConfig instances with different values hash differently."""
    mc1 = MarketConfig(market_id="EOM")
    mc2 = MarketConfig(market_id="OTHER")
    assert hash(mc1) != hash(mc2)


@pytest.fixture
async def market_role() -> MarketRole:
    market_id = "Test"
    marketconfig = MarketConfig(
        market_id=market_id,
        opening_hours=rr.rrule(rr.HOURLY, dtstart=start, until=end),
        opening_duration=rd(hours=1),
        market_mechanism="pay_as_clear",
        market_products=[MarketProduct(rd(hours=1), 1, rd(hours=1))],
    )
    clock = ExternalClock(0)
    container = create_tcp_container(addr=("0.0.0.0", 9098), clock=clock)
    market_agent = RoleAgent()
    market_role = MarketRole(marketconfig=marketconfig)
    market_agent.add_role(market_role)
    container.register(market_agent)

    async with activate(container):
        yield market_role

        end_ts = datetime2timestamp(end)
        clock.set_time(end_ts)
        await tasks_complete_or_sleeping(container)


async def test_market_init(market_role: MarketRole):
    meta = {
        "sender_addr": market_role.context.addr,
        "sender_id": market_role.context.aid,
    }
    end = start + rd(hours=1)
    orderbook = [
        {
            "start_time": start,
            "end_time": end,
            "volume": 120,
            "price": 120,
            "agent_addr": "gen1",
            "only_hours": None,
        }
    ]
    market_role.open_auctions |= {(start, end, None)}
    market_role.handle_orderbook(content={"orderbook": orderbook}, meta=meta)
    assert len(market_role.all_orders) == 1


async def test_market_tick(market_role: MarketRole):
    meta = {
        "sender_addr": market_role.context.addr,
        "sender_id": market_role.context.aid,
    }
    market_role.marketconfig.price_tick = 0.1

    orderbook = [
        {
            "start_time": start,
            "end_time": end,
            "volume": 120.123,
            "price": 1201,  # used to present 120.1 with precision 0.1
            "agent_addr": "gen1",
            "only_hours": None,
        }
    ]
    market_role.open_auctions |= {(start, end, None)}
    market_role.handle_orderbook(content={"orderbook": orderbook}, meta=meta)
    assert len(market_role.all_orders) == 1
    assert market_role.all_orders[0]["price"] == 1201

    orderbook = [
        {
            "start_time": start,
            "end_time": end,
            "volume": 120.123,
            "price": 120.123,
            "agent_addr": "gen1",
            "only_hours": None,
        }
    ]

    market_role.handle_orderbook(content={"orderbook": orderbook}, meta=meta)
    # this does not work
    assert len(market_role.all_orders) == 1


async def test_market_max(market_role: MarketRole):
    meta = {
        "sender_addr": market_role.context.addr,
        "sender_id": market_role.context.aid,
    }
    market_role.marketconfig.maximum_bid_price = 1000
    market_role.marketconfig.minimum_bid_price = -500
    market_role.marketconfig.maximum_bid_volume = 9090
    market_role.open_auctions |= {(start, end, None)}

    orderbook = [
        {
            "start_time": start,
            "end_time": end,
            "volume": 9091,
            "price": 120,
            "agent_addr": "gen1",
            "only_hours": None,
        }
    ]
    market_role.handle_orderbook(content={"orderbook": orderbook}, meta=meta)
    # assert if the volume was reduced to the maximum
    assert market_role.all_orders[0]["volume"] == 9090

    orderbook = [
        {
            "start_time": start,
            "end_time": end,
            "volume": 9090,
            "price": 1001,
            "agent_addr": "gen1",
            "only_hours": None,
        }
    ]
    market_role.handle_orderbook(content={"orderbook": orderbook}, meta=meta)
    # assert if the price was reduced to the maximum
    assert market_role.all_orders[1]["price"] == 1000

    orderbook = [
        {
            "start_time": start,
            "end_time": end,
            "volume": 9090,
            "price": -550,
            "agent_addr": "gen1",
            "only_hours": None,
        }
    ]
    market_role.handle_orderbook(content={"orderbook": orderbook}, meta=meta)
    # assert if the price was increased to the minimum
    assert market_role.all_orders[2]["price"] == -500

    orderbook = [
        {
            "start_time": start,
            "end_time": end,
            "volume": 9090,
            "price": 1000,
            "agent_addr": "gen1",
            "only_hours": None,
        }
    ]
    market_role.handle_orderbook(content={"orderbook": orderbook}, meta=meta)
    assert len(market_role.all_orders) == 4
    assert market_role.all_orders[3]["price"] == 1000
    assert market_role.all_orders[3]["volume"] == 9090


async def test_market_for_BB(market_role: MarketRole):
    meta = {
        "sender_addr": market_role.context.addr,
        "sender_id": market_role.context.aid,
    }
    market_role.marketconfig.maximum_bid_price = 1000
    market_role.marketconfig.minimum_bid_price = -500
    market_role.marketconfig.maximum_bid_volume = 9090

    end = start + rd(hours=24)
    time_range = pd.date_range(start, end - pd.Timedelta("1h"), freq="1h")
    market_role.open_auctions |= {
        (time, time + rd(hours=1), None) for time in time_range
    }

    orderbook = [
        {
            "start_time": start,
            "end_time": end,
            "volume": {time: 50 for time in time_range},
            "price": {time: 1000 for time in time_range},
            "agent_addr": "gen1",
            "only_hours": None,
            "bid_type": "BB",
        }
    ]
    market_role.handle_orderbook(content={"orderbook": orderbook}, meta=meta)
    assert len(market_role.all_orders) == 1


async def test_market_registration(market_role: MarketRole):
    meta = {
        "sender_addr": "test_address",
        "sender_id": "test_aid",
    }

    assert market_role.registered_agents == {}
    info = [{"technology": "nuclear", "max_power": 2}]
    market_role.handle_registration(
        {"market_id": market_role.marketconfig.market_id, "information": info},
        meta=meta,
    )
    assert len(market_role.registered_agents.keys()) == 1
    assert market_role.registered_agents[sender_addr(meta)] == info


async def test_market_unmatched(market_role: MarketRole):
    meta = {
        "sender_addr": "test_address",
        "sender_id": "test_aid",
    }

    orderbook = [
        {
            "start_time": start,
            "end_time": end,
            "volume": 10,
            "price": 20,
            "agent_addr": "gen1",
            "only_hours": None,
        }
    ]
    market_role.open_auctions |= {(start, end, None)}
    market_role.handle_orderbook(content={"orderbook": orderbook}, meta=meta)

    content = {
        "order": {
            "start_time": start,
            "end_time": start + rd(hours=1),
            "only_hours": None,
        }
    }

    market_role.handle_get_unmatched(content, meta)


async def test_market_accepted(market_role: MarketRole):
    meta = {
        "sender_addr": "test_address",
        "sender_id": "test_aid",
    }

    orderbook = [
        {
            "start_time": start,
            "end_time": end,
            "volume": 10,
            "price": 20,
            "agent_addr": "gen1",
            "only_hours": None,
        },
        {
            "start_time": start,
            "end_time": end,
            "volume": -10,
            "price": 20,
            "agent_addr": "gen1",
            "only_hours": None,
        },
    ]
    market_role.open_auctions |= {(start, end, None)}
    market_role.handle_orderbook(content={"orderbook": orderbook}, meta=meta)

    content = {
        "order": {
            "start_time": start,
            "end_time": start + rd(hours=1),
            "only_hours": None,
        }
    }

    def accept_clear(all_orders, products):
        return all_orders, [], [{"price": 0, "product_start": products[0][0]}], None

    market_role.clear = accept_clear
    # todo add exception handler

    accepted, meta = await market_role.clear_market([(start, end, None)])
    assert accepted == orderbook


def _plain_market_role():
    marketconfig = MarketConfig(
        market_id="Test",
        opening_hours=rr.rrule(rr.HOURLY, dtstart=start, until=end),
        opening_duration=rd(hours=1),
        market_mechanism="pay_as_clear",
        market_products=[MarketProduct(rd(hours=1), 1, rd(hours=1))],
        maximum_bid_price=1000,
        minimum_bid_price=-500,
        maximum_bid_volume=100,
    )
    return MarketRole(marketconfig=marketconfig)


def test_validate_orderbook_drops_order_for_closed_product():
    role = _plain_market_role()
    open_end = start + rd(hours=1)
    role.open_auctions = {(start, open_end, None)}
    orderbook = [
        {"start_time": start, "end_time": open_end, "volume": 10, "price": 5},
        {
            "start_time": open_end,
            "end_time": open_end + rd(hours=1),
            "volume": 10,
            "price": 5,
        },
    ]

    role.validate_orderbook(orderbook, "agent")

    assert len(orderbook) == 1
    assert orderbook[0]["start_time"] == start


def test_validate_orderbook_clips_block_bid_values():
    role = _plain_market_role()
    bb_end = start + rd(hours=3)
    hours = pd.date_range(start, bb_end - pd.Timedelta("1h"), freq="1h")
    role.open_auctions = {(h, h + rd(hours=1), None) for h in hours}
    orderbook = [
        {
            "start_time": start,
            "end_time": bb_end,
            "volume": {h: v for h, v in zip(hours, [50, 150, -200])},
            "price": {h: p for h, p in zip(hours, [1500, 10, -900])},
            "bid_type": "BB",
            "bid_id": "bb1",
        }
    ]

    role.validate_orderbook(orderbook, "agent")

    assert len(orderbook) == 1
    assert list(orderbook[0]["price"].values()) == [1000, 10, -500]
    assert list(orderbook[0]["volume"].values()) == [50, 100, -100]


def test_validate_orderbook_drops_block_bid_with_closed_hour_and_children():
    role = _plain_market_role()
    bb_end = start + rd(hours=2)
    hours = pd.date_range(start, bb_end - pd.Timedelta("1h"), freq="1h")
    # only the first hour of the block is open
    role.open_auctions = {(start, start + rd(hours=1), None)}
    orderbook = [
        {
            "start_time": start,
            "end_time": bb_end,
            "volume": dict.fromkeys(hours, 10),
            "price": dict.fromkeys(hours, 10),
            "bid_type": "BB",
            "bid_id": "parent",
        },
        {
            "start_time": start,
            "end_time": start + rd(hours=1),
            "volume": {start: 5},
            "price": {start: 10},
            "bid_type": "LB",
            "bid_id": "child",
            "parent_bid_id": "parent",
        },
    ]

    role.validate_orderbook(orderbook, "agent")

    assert orderbook == []
