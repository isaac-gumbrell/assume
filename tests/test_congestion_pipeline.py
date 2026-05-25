# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Integration tests for the two-stage congestion pipeline:

Stage 1: Run a naive nodal clearing simulation → produces grid_flows
Stage 2: Convert grid_flows → congestion_df.csv → feed to learning agents

These tests verify:
- flows_to_congestion_df() correctly converts grid flow output to directional signals
- The naive scenario inputs load successfully
- The learning scenario inputs load successfully (when congestion_df.csv is present)
- The full conversion pipeline produces valid congestion signals
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from assume.common.grid_utils import flows_to_congestion_df

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SCENARIO_DIR = Path(__file__).parent / "inputs"
NAIVE_SCENARIO = "congestion_scenario_naive"
LEARNING_SCENARIO = "congestion_scenario_learning"


@pytest.fixture
def lines_df():
    """Load the shared lines.csv from the test scenario."""
    return pd.read_csv(SCENARIO_DIR / NAIVE_SCENARIO / "lines.csv", index_col=0)


@pytest.fixture
def sample_grid_flows():
    """Create a synthetic grid_flows DataFrame resembling OutputRole.convert_flows() output.

    Simulates 24h of flow on three lines (Line_N_C, Line_C_S, Line_N_S)
    with the pattern: strong north→south flow during peak hours,
    moderate reverse flow at night.  This mimics reality where cheap
    northern generation exports to the south during the day.
    """
    timestamps = pd.date_range("2019-01-01", periods=24, freq="h")
    rows = []
    for i, ts in enumerate(timestamps):
        hour = ts.hour
        # Peak hours: strong N→S flow
        if 6 <= hour <= 20:
            flow_n_c = 400.0 + 50 * np.sin(np.pi * (hour - 6) / 14)
            flow_c_s = 350.0 + 30 * np.sin(np.pi * (hour - 6) / 14)
            flow_n_s = 250.0 + 40 * np.sin(np.pi * (hour - 6) / 14)
        else:
            # Night: moderate reverse flow (south→north)
            flow_n_c = -100.0
            flow_c_s = -80.0
            flow_n_s = -50.0

        rows.append({"datetime": ts, "line": "Line_N_C", "flow": flow_n_c})
        rows.append({"datetime": ts, "line": "Line_C_S", "flow": flow_c_s})
        rows.append({"datetime": ts, "line": "Line_N_S", "flow": flow_n_s})

    df = pd.DataFrame(rows).set_index("datetime")
    df["simulation"] = "congestion_scenario_naive_naive_baseline"
    return df


# ---------------------------------------------------------------------------
# Tests: flows_to_congestion_df
# ---------------------------------------------------------------------------


class TestFlowsToCongestionDf:
    """Tests for the flows → congestion signal conversion utility."""

    def test_basic_conversion_shape(self, sample_grid_flows, lines_df):
        """Output has one signal column per line, same length as input."""
        congestion = flows_to_congestion_df(sample_grid_flows, lines_df)
        assert len(congestion) == 24
        assert "Line_N_C_line_congestion_signal" in congestion.columns
        assert "Line_C_S_line_congestion_signal" in congestion.columns
        assert "Line_N_S_line_congestion_signal" in congestion.columns

    def test_signals_are_bounded(self, sample_grid_flows, lines_df):
        """All congestion signals must be in [-1, +1]."""
        congestion = flows_to_congestion_df(sample_grid_flows, lines_df)
        for col in congestion.columns:
            assert congestion[col].min() >= -1.0, f"{col} below -1"
            assert congestion[col].max() <= 1.0, f"{col} above +1"

    def test_directional_asymmetry(self, sample_grid_flows, lines_df):
        """Forward and reverse signals reflect asymmetric NTC capacities.

        Line_N_C: s_nom_forward=500, s_nom_reverse=300
        A flow of -100 MW (reverse) should give signal -100/300 = -0.333
        A flow of +400 MW (forward) should give signal 400/500 = 0.8
        """
        congestion = flows_to_congestion_df(sample_grid_flows, lines_df)
        signal = congestion["Line_N_C_line_congestion_signal"]

        # Night hours (0-5, 21-23) have flow = -100 → signal = -0.333
        # flow=-100, cap_reverse=300: flow / -300 = 0.333, then *-1 = -0.333
        night_signals = signal.iloc[:6]  # hours 0-5
        np.testing.assert_allclose(night_signals.values, -1.0 / 3.0, atol=0.01)

    def test_peak_flow_gives_high_signal(self, sample_grid_flows, lines_df):
        """Peak hour flows should produce congestion signals near line capacity."""
        congestion = flows_to_congestion_df(sample_grid_flows, lines_df)
        signal = congestion["Line_N_C_line_congestion_signal"]

        # Hours 6-20 have flow ~400-450, cap_forward=500 → signal ~0.8-0.9
        peak_signal = signal.iloc[6:21]
        assert peak_signal.min() > 0.5, "Peak signals should be high"
        assert peak_signal.max() <= 1.0, "Peak signals should not exceed 1.0"

    def test_empty_inputs(self, lines_df):
        """Empty flows DataFrame returns empty result."""
        empty = pd.DataFrame(columns=["line", "flow"])
        empty.index.name = "datetime"
        result = flows_to_congestion_df(empty, lines_df)
        assert result.empty

    def test_zero_flow_gives_zero_signal(self, lines_df):
        """Zero flow should produce zero congestion signal."""
        ts = pd.date_range("2019-01-01", periods=3, freq="h")
        flows = pd.DataFrame(
            {
                "line": ["Line_N_C"] * 3,
                "flow": [0.0, 0.0, 0.0],
            },
            index=ts,
        )
        congestion = flows_to_congestion_df(flows, lines_df)
        np.testing.assert_allclose(
            congestion["Line_N_C_line_congestion_signal"].values, 0.0
        )

    def test_clipping_at_overcapacity(self, lines_df):
        """Flows exceeding line capacity are clipped to ±1."""
        ts = pd.date_range("2019-01-01", periods=2, freq="h")
        flows = pd.DataFrame(
            {
                "line": ["Line_N_C", "Line_N_C"],
                "flow": [1000.0, -600.0],  # exceed both forward (500) and reverse (300)
            },
            index=ts,
        )
        congestion = flows_to_congestion_df(flows, lines_df)
        signal = congestion["Line_N_C_line_congestion_signal"].values
        assert signal[0] == 1.0, "Forward overflow should clip to +1"
        assert signal[1] == -1.0, "Reverse overflow should clip to -1"


# ---------------------------------------------------------------------------
# Tests: Scenario loading
# ---------------------------------------------------------------------------


class TestNaiveScenarioInputs:
    """Verify that the naive scenario input files are well-formed and loadable."""

    def test_buses_csv_loads(self):
        buses = pd.read_csv(SCENARIO_DIR / NAIVE_SCENARIO / "buses.csv", index_col=0)
        assert set(buses.index) == {"north", "centre", "south"}
        assert "zone_id" in buses.columns

    def test_lines_csv_has_directional_ntc(self, lines_df):
        assert "s_nom_forward" in lines_df.columns
        assert "s_nom_reverse" in lines_df.columns
        # All lines should have forward > reverse (asymmetric)
        for line_id in lines_df.index:
            assert (
                lines_df.loc[line_id, "s_nom_forward"]
                > lines_df.loc[line_id, "s_nom_reverse"]
            )

    def test_powerplant_units_have_correct_bidding(self):
        pp = pd.read_csv(
            SCENARIO_DIR / NAIVE_SCENARIO / "powerplant_units.csv", index_col=0
        )
        assert "bidding_EOM" in pp.columns
        assert (pp["bidding_EOM"] == "powerplant_energy_naive").all()

    def test_demand_covers_three_nodes(self):
        demand = pd.read_csv(
            SCENARIO_DIR / NAIVE_SCENARIO / "demand_df.csv",
            index_col=0,
            parse_dates=True,
        )
        assert "demand_north" in demand.columns
        assert "demand_centre" in demand.columns
        assert "demand_south" in demand.columns

    def test_south_demand_exceeds_local_generation(self):
        """South demand should exceed local generation to force north→south flows."""
        demand = pd.read_csv(
            SCENARIO_DIR / NAIVE_SCENARIO / "demand_df.csv",
            index_col=0,
            parse_dates=True,
        )
        pp = pd.read_csv(
            SCENARIO_DIR / NAIVE_SCENARIO / "powerplant_units.csv", index_col=0
        )
        south_gen = pp.loc[pp["node"] == "south", "max_power"].sum()
        south_peak_demand = demand["demand_south"].max()
        assert south_peak_demand > south_gen, (
            f"South peak demand ({south_peak_demand}) should exceed "
            f"south generation capacity ({south_gen}) to create congestion"
        )


class TestLearningScenarioInputs:
    """Verify that the learning scenario input files are well-formed."""

    def test_powerplant_units_use_learning_strategies(self):
        pp = pd.read_csv(
            SCENARIO_DIR / LEARNING_SCENARIO / "powerplant_units.csv", index_col=0
        )
        assert "bidding_EOM" in pp.columns
        assert (pp["bidding_EOM"] == "powerplant_energy_learning_single_bid").all()

    def test_storage_units_use_learning_strategy(self):
        storage = pd.read_csv(
            SCENARIO_DIR / LEARNING_SCENARIO / "storage_units.csv", index_col=0
        )
        assert "bidding_EOM" in storage.columns
        assert (storage["bidding_EOM"] == "storage_energy_learning").all()

    def test_config_has_congestion_observation_enabled(self):
        import yaml

        with open(SCENARIO_DIR / LEARNING_SCENARIO / "config.yaml") as f:
            config = yaml.safe_load(f)

        learning_case = config["learning_congestion"]
        params = learning_case["bidding_strategy_params"]
        assert params["include_local_line_congestion_observation"] is True
        assert params["include_price_forecast_observation"] is False
        assert params["include_price_history_observation"] is False


# ---------------------------------------------------------------------------
# Tests: End-to-end pipeline conversion
# ---------------------------------------------------------------------------


class TestCongestionPipeline:
    """Test the complete flow: grid_flows → congestion_df.csv → loadable by scenario."""

    def test_write_and_read_congestion_csv(self, sample_grid_flows, lines_df, tmp_path):
        """Convert flows to congestion, save as CSV, and verify it can be re-loaded."""
        congestion = flows_to_congestion_df(sample_grid_flows, lines_df)

        # Write to the learning scenario format
        csv_path = tmp_path / "congestion_df.csv"
        congestion.to_csv(csv_path)

        # Re-read
        loaded = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        assert set(loaded.columns) == set(congestion.columns)
        np.testing.assert_allclose(loaded.values, congestion.values, atol=1e-6)

    def test_congestion_df_columns_match_forecaster_expectations(
        self, sample_grid_flows, lines_df
    ):
        """Congestion column names should end with '_line_congestion_signal'
        which is what the UnitForecaster looks for when loading congestion data."""
        congestion = flows_to_congestion_df(sample_grid_flows, lines_df)
        for col in congestion.columns:
            assert col.endswith("_line_congestion_signal"), (
                f"Column {col} doesn't match expected naming"
            )

    def test_congestion_signal_correlates_with_demand_imbalance(self):
        """Verify that congestion signals are higher when south demand exceeds local supply.

        This is the economic intuition test: when the south needs imports,
        north→south lines become congested (positive signal).
        """
        lines_df = pd.read_csv(SCENARIO_DIR / NAIVE_SCENARIO / "lines.csv", index_col=0)
        demand_df = pd.read_csv(
            SCENARIO_DIR / NAIVE_SCENARIO / "demand_df.csv",
            index_col=0,
            parse_dates=True,
        )

        # Create flows that are proportional to south demand excess
        pp = pd.read_csv(
            SCENARIO_DIR / NAIVE_SCENARIO / "powerplant_units.csv", index_col=0
        )
        south_gen = pp.loc[pp["node"] == "south", "max_power"].sum()

        timestamps = demand_df.index[:24]
        rows = []
        for ts in timestamps:
            south_excess = max(0, demand_df.loc[ts, "demand_south"] - south_gen)
            # Lines carry some fraction of the excess
            rows.append(
                {"datetime": ts, "line": "Line_N_C", "flow": south_excess * 0.4}
            )
            rows.append(
                {"datetime": ts, "line": "Line_C_S", "flow": south_excess * 0.35}
            )
            rows.append(
                {"datetime": ts, "line": "Line_N_S", "flow": south_excess * 0.25}
            )

        flows = pd.DataFrame(rows).set_index("datetime")
        congestion = flows_to_congestion_df(flows, lines_df)

        # At low demand, flow ~ 0 → signal ~ 0
        # At peak demand, flow is large → signal should be positive
        signal = congestion["Line_N_C_line_congestion_signal"]
        peak_idx = demand_df["demand_south"][:24].idxmax()
        trough_idx = demand_df["demand_south"][:24].idxmin()

        assert signal.loc[peak_idx] > signal.loc[trough_idx], (
            "Congestion should be higher at peak south demand"
        )

    def test_read_congestion_from_results_utility(
        self, sample_grid_flows, lines_df, tmp_path
    ):
        """Test the convenience function read_congestion_from_results."""
        from assume.common.grid_utils import read_congestion_from_results

        # Set up directory structure mimicking output + network
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()
        network_dir = tmp_path / "network"
        network_dir.mkdir()

        # Write grid_flows.csv (from OutputRole)
        sample_grid_flows.to_csv(output_dir / "grid_flows.csv")

        # Write lines.csv
        lines_df.to_csv(network_dir / "lines.csv")

        # Call the utility
        congestion = read_congestion_from_results(output_dir, network_dir)

        assert len(congestion) == 24
        assert all(
            col.endswith("_line_congestion_signal") for col in congestion.columns
        )
        assert congestion.min().min() >= -1.0
        assert congestion.max().max() <= 1.0
