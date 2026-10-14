# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, event, inspect, text

from assume.common.outputs import WriteOutput

os.makedirs("./examples/local_db", exist_ok=True)
DB_URI = "sqlite:///./examples/local_db/test_outputs.db"


def test_output_market_orders():
    engine = create_engine(DB_URI)
    start = datetime(2020, 1, 1)
    end = datetime(2020, 1, 2)
    output_writer = WriteOutput("test_sim", start, end, engine)
    assert len(output_writer.write_buffers.keys()) == 0
    meta = {"sender_id": None}
    content = {
        "context": "write_results",
        "type": "market_orders",
        "sender": "CRM_pos",
        "data": [],
    }
    output_writer.handle_output_message(content, meta)
    assert len(output_writer.write_buffers["market_orders"]) == 0

    orderbook = [
        {
            "start_time": start,
            "end_time": end,
            "volume": 120,
            "price": 120,
            "agent_addr": "gen1",
            "only_hours": None,
        },
        {
            "start_time": start,
            "end_time": end,
            "volume": 80,
            "price": 58,
            "agent_addr": "gen1",
            "only_hours": None,
        },
        {
            "start_time": start,
            "end_time": end,
            "volume": 100,
            "price": 53,
            "agent_addr": "gen1",
            "only_hours": None,
        },
        {
            "start_time": start,
            "end_time": end,
            "volume": -180,
            "price": 70,
            "agent_addr": "dem1",
            "only_hours": None,
        },
    ]

    content = {
        "context": "write_results",
        "type": "market_orders",
        "sender": "CRM_pos",
        "data": orderbook,
    }
    output_writer.handle_output_message(content, meta)
    assert len(output_writer.write_buffers["market_orders"]) == 1


def test_output_market_results():
    engine = create_engine(DB_URI)
    start = datetime(2020, 1, 1)
    end = datetime(2020, 1, 2)
    output_writer = WriteOutput("test_sim", start, end, engine)
    assert len(output_writer.write_buffers.keys()) == 0
    meta = {"sender_id": None}
    content = {
        "context": "write_results",
        "type": "market_meta",
        "sender": "CRM_pos",
        "data": [
            {
                "supply_volume": 0,
                "demand_volume": 0,
                "demand_volume_energy": 0.0,
                "supply_volume_energy": 0.0,
                "price": 0.0,
                "max_price": 0,
                "min_price": 0,
                "node": None,
                "product_start": datetime(2019, 1, 1, 2),
                "product_end": datetime(2019, 1, 1, 6),
                "only_hours": None,
                "market_id": "CRM_pos",
                "time": 1546302600,
            }
        ],
    }
    output_writer.handle_output_message(content, meta)
    assert len(output_writer.write_buffers["market_meta"]) == 1, "market_meta"


def test_output_market_dispatch():
    engine = create_engine(DB_URI)
    start = datetime(2020, 1, 1)
    end = datetime(2020, 1, 2)
    output_writer = WriteOutput("test_sim", start, end, engine)
    assert len(output_writer.write_buffers.keys()) == 0
    meta = {"sender_id": None}
    content = {"context": "write_results", "type": "market_dispatch", "data": []}
    output_writer.handle_output_message(content, meta)
    # empty dfs are discarded
    assert len(output_writer.write_buffers["market_dispatch"]) == 0, "market_dispatch"

    content = {
        "context": "write_results",
        "type": "market_dispatch",
        "data": [[start, 90, "EOM", "TestUnit"]],
    }
    output_writer.handle_output_message(content, meta)
    assert len(output_writer.write_buffers["market_dispatch"]) == 1, "market_dispatch"


def test_output_unit_dispatch():
    engine = create_engine(DB_URI)
    start = datetime(2020, 1, 1)
    end = datetime(2020, 1, 2)
    output_writer = WriteOutput("test_sim", start, end, engine)
    assert len(output_writer.write_buffers.keys()) == 0
    meta = {"sender_id": None}
    content = {
        "context": "write_results",
        "type": "unit_dispatch",
        "data": [
            {
                "power": np.array([0.0, 1000.0]),
                "energy_cashflow": np.array([0.0, 45050.0]),
                "time": [datetime(2022, 1, 1, 0), datetime(2022, 1, 1, 1)],
                "unit": "Unit 2",
            }
        ],
    }

    output_writer.handle_output_message(content, meta)
    assert len(output_writer.write_buffers["unit_dispatch"]) == 1, "unit_dispatch"


def test_output_write_flows():
    engine = create_engine(DB_URI)
    start = datetime(2020, 1, 1)
    end = datetime(2020, 1, 2)
    output_writer = WriteOutput("test_sim", start, end, engine)
    assert len(output_writer.write_buffers.keys()) == 0
    meta = {"sender_id": None}
    content = {
        "context": "write_results",
        "type": "grid_flows",
        "data": {(datetime(2019, 1, 1, 0, 0), "north_south_example"): 0.0},
    }

    output_writer.handle_output_message(content, meta)
    assert len(output_writer.write_buffers["grid_flows"]) == 1, "grid_flows"


def _make_writer(db_uri, simulation_id="sim_a", **kwargs):
    writer = WriteOutput(
        simulation_id=simulation_id,
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 2),
        save_frequency_hours=None,
        db_uri=db_uri,
        **kwargs,
    )
    writer.db = create_engine(db_uri)
    return writer


def _seed_tables(engine):
    pd.DataFrame({"price": [10.0, 20.0], "simulation": ["sim_a", "sim_b"]}).to_sql(
        "market_meta", engine, index=False
    )
    pd.DataFrame(
        {
            "reward": [1.0, 2.0],
            "episode": [1, 1],
            "simulation": ["sim_a", "sim_b"],
        }
    ).to_sql("rl_params", engine, index=False)


def _simulations(engine, table):
    with engine.connect() as conn:
        rows = conn.execute(text(f"select simulation from {table}")).fetchall()
    return sorted(row[0] for row in rows)


def test_delete_db_scenario_only_removes_own_simulation(tmp_path):
    db_uri = f"sqlite:///{tmp_path / 'test.db'}"
    writer = _make_writer(db_uri)
    _seed_tables(writer.db)

    writer.delete_db_scenario("sim_a")

    assert _simulations(writer.db, "market_meta") == ["sim_b"]
    # rl tables are only cleared in the first learning episode
    assert _simulations(writer.db, "rl_params") == ["sim_a", "sim_b"]

    inspector = inspect(writer.db)
    market_indexes = {i["name"] for i in inspector.get_indexes("market_meta")}
    rl_indexes = {i["name"] for i in inspector.get_indexes("rl_params")}
    assert "market_meta_scenario" in market_indexes
    assert {"rl_params_scenario", "rl_params_simulation_episode"} <= rl_indexes
    writer.db.dispose()


def test_delete_db_scenario_clears_rl_tables_in_first_learning_episode(tmp_path):
    db_uri = f"sqlite:///{tmp_path / 'test.db'}"
    writer = _make_writer(db_uri, learning_mode=True, episode=1)
    _seed_tables(writer.db)

    writer.delete_db_scenario("sim_a")

    assert _simulations(writer.db, "market_meta") == ["sim_b"]
    assert _simulations(writer.db, "rl_params") == ["sim_b"]
    writer.db.dispose()


def test_delete_db_scenario_keeps_index_creation_out_of_delete(tmp_path):
    db_uri = f"sqlite:///{tmp_path / 'test.db'}"
    writer = _make_writer(db_uri)
    _seed_tables(writer.db)
    # first call creates the missing indexes
    writer.delete_db_scenario("sim_a")

    statements = []

    @event.listens_for(writer.db, "before_cursor_execute")
    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement.lower())

    writer.delete_db_scenario("sim_b")

    # existing indexes must not be re-created, as CREATE INDEX locks the
    # whole table against inserts of concurrent simulations on PostgreSQL
    assert not any("create index" in s for s in statements)
    assert any(s.startswith("delete from") for s in statements)
    assert _simulations(writer.db, "market_meta") == []
    writer.db.dispose()


async def test_on_stop_stores_kpis_and_releases_connections(tmp_path):
    db_uri = f"sqlite:///{tmp_path / 'test.db'}"
    writer = _make_writer(db_uri)
    writer.handle_output_message(
        {
            "context": "write_results",
            "type": "market_meta",
            "data": [
                {
                    "price": 50.0,
                    "demand_volume_energy": 100.0,
                    "market_id": "EOM",
                    "time": datetime(2020, 1, 1),
                }
            ],
        },
        {"sender_id": None},
    )

    await writer.on_stop()

    # no connection may be left in the pool after the engine was disposed
    assert writer.db.pool.checkedin() == 0
    assert writer.db.pool.checkedout() == 0

    kpis = pd.read_sql("select * from kpis", writer.db)
    writer.db.dispose()
    avg_price = kpis.loc[kpis["variable"] == "avg_price", "value"]
    assert avg_price.tolist() == [50.0]


def test_get_sum_reward_releases_connection(tmp_path):
    db_uri = f"sqlite:///{tmp_path / 'test.db'}"
    writer = _make_writer(db_uri)
    pd.DataFrame(
        {
            "unit": ["u1", "u1", "u2"],
            "reward": [1.0, 2.0, 5.0],
            "evaluation_mode": [True, True, True],
            "episode": [1, 1, 1],
            "simulation": ["sim_a", "sim_a", "sim_a"],
        }
    ).to_sql("rl_params", writer.db, index=False)
    writer.db.dispose()

    rewards = writer.get_sum_reward(episode=1)

    assert sorted(rewards.tolist()) == [3.0, 5.0]
    assert writer.db.pool.checkedin() == 0


def _dispatch_rows(hour):
    return [
        {
            "datetime": datetime(2020, 1, 1, hour),
            "power": 10.0,
            "market_id": "EOM",
            "unit_id": "u1",
        }
    ]


async def test_store_dfs_retries_failed_db_write_without_duplicates(tmp_path):
    db_uri = f"sqlite:///{tmp_path / 'test.db'}"
    writer = _make_writer(db_uri, export_csv_path=str(tmp_path / "csv"))
    original_write = writer._write_df_to_db
    calls = {"n": 0}

    def flaky_write(table, df):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("database unavailable")
        original_write(table, df)

    writer._write_df_to_db = flaky_write

    writer.write_buffers["market_dispatch"].extend(_dispatch_rows(0))
    await writer.store_dfs()
    # the failed batch is kept, the csv has already been written
    assert len(writer._pending_writes["market_dispatch"]) == 1
    csv_path = writer.export_csv_path / "market_dispatch.csv"
    assert len(pd.read_csv(csv_path)) == 1

    writer.write_buffers["market_dispatch"].extend(_dispatch_rows(1))
    await writer.store_dfs()

    assert not writer._pending_writes["market_dispatch"]
    # each row is written exactly once to each destination
    assert len(pd.read_csv(csv_path)) == 2
    db_rows = pd.read_sql("select * from market_dispatch", writer.db)
    assert sorted(pd.to_datetime(db_rows["datetime"]).dt.hour.tolist()) == [0, 1]
    writer.db.dispose()


def test_store_kpis_uses_eval_episode_in_evaluation_mode(tmp_path):
    db_uri = f"sqlite:///{tmp_path / 'test.db'}"
    writer = _make_writer(
        db_uri,
        learning_mode=True,
        evaluation_mode=True,
        episode=5,
        eval_episode=2,
    )
    pd.DataFrame(
        {
            "unit": ["u1", "u1", "u1"],
            "reward": [1.0, 2.0, 100.0],
            "regret": [0.0, 0.0, 0.0],
            "profit": [1.0, 2.0, 100.0],
            "evaluation_mode": [True, True, False],
            "episode": [2, 2, 2],
            "simulation": ["sim_a", "sim_a", "sim_a"],
        }
    ).to_sql("rl_params", writer.db, index=False)

    writer.store_kpis()

    kpis = pd.read_sql("select * from kpis", writer.db).set_index("variable")
    # only evaluation rows of eval episode 2 are counted
    assert kpis.loc["sum_reward", "value"] == 3.0
    assert kpis.loc["sum_profit", "value"] == 3.0
    writer.db.dispose()
