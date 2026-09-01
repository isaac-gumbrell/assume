# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import copy
import json
import logging
import os
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import dateutil.rrule as rr
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from assume.common.exceptions import AssumeException
from assume.common.fast_pandas import FastIndex
from assume.common.forecaster import (
    BuildingForecaster,
    CustomUnitForecaster,
    DemandForecaster,
    DsmUnitForecaster,
    ExchangeForecaster,
    HydrogenForecaster,
    PowerplantForecaster,
    SteamgenerationForecaster,
    SteelplantForecaster,
    UnitForecaster,
)
from assume.common.market_objects import MarketConfig, MarketProduct
from assume.common.utils import (
    adjust_unit_operator_for_learning,
    confirm_learning_save_path,
    convert_to_rrule_freq,
    load_index_file,
    normalize_availability,
    set_random_seed,
)
from assume.strategies import BaseStrategy
from assume.world import World

logger = logging.getLogger(__name__)


def bidding_strategies_from_param_dict(param_dict: dict):
    return {
        ident.split("bidding_")[1]: strategy
        for ident, strategy in param_dict.items()
        if ident.startswith("bidding_")
    }


def forecast_algorithm_from_param_dict(param_dict: dict) -> dict[str, str]:
    return {
        ident.split("forecast_")[1]: forecast_algorithm
        for ident, forecast_algorithm in param_dict.items()
        if ident.startswith("forecast_")
    }


def get_unit_forecast_algorithms(
    forecast_algorithms: dict[str, str], plant: dict
) -> dict[str, str]:
    unit_forecast_algorithms = forecast_algorithm_from_param_dict(
        plant
    )  # get forecast specific parts

    # overwrite None in plant csv with values from config if it exists!
    for key, forecast_alg in unit_forecast_algorithms.items():
        if forecast_alg is None or pd.isna(forecast_alg):
            unit_forecast_algorithms[key] = forecast_algorithms.get(key)

    return forecast_algorithms | unit_forecast_algorithms  # merge dicts together


def load_file(
    path: str,
    config: dict,
    file_name: str,
    index: pd.DatetimeIndex | None = None,
    check_duplicates: bool = True,
) -> pd.DataFrame:
    """
    Loads a csv file from the given path and returns a dataframe.

    The config file is used to check if the file name is specified in the config file,
    otherwise defaults to the file name.

    If the index is specified, the dataframe is resampled to the index, if possible. If not, None is returned.

    Args:
        path (str): The path to the csv file.
        config (dict): The config file containing file mappings.
        file_name (str): The name of the csv file.
        index (pd.DatetimeIndex, optional): The index of the dataframe. Defaults to None.
        check_duplicates (bool, optional): Whether to check for duplicate unit names. Defaults to True.

    Returns:
        pandas.DataFrame: The dataframe containing the loaded data.

    Raises:
        FileNotFoundError: If the specified file is not found, returns None.
    """
    if file_name in config:
        if config[file_name] is None:
            return None
        file_path = Path(path) / config[file_name]
    else:
        file_path = Path(path) / f"{file_name}.csv"

    try:
        if index is not None:
            df = load_index_file(file_path, index)
        else:
            df = pd.read_csv(
                file_path,
                index_col=0,
                encoding="utf-8",
                na_values=["n.a.", "None", "-", "none", "nan"],
                parse_dates=index is not None,
            )
            for col in df:
                # check if the column is of dtype int
                if df[col].dtype == "int":
                    # convert the column to float
                    df[col] = df[col].astype(float)

            if check_duplicates:
                # Check if duplicate unit names exist and raise an error
                duplicates = df.index[df.index.duplicated()].unique()

                if len(duplicates) > 0:
                    duplicate_names = ", ".join(map(str, duplicates))
                    raise ValueError(
                        f"Duplicate unit names found in {file_name}: {duplicate_names}. Please rename them to avoid conflicts."
                    )
        return df

    except FileNotFoundError:
        logger.info(f"{file_path} not found. Returning None")
        return None


def load_dsm_units(
    path: str,
    config: dict,
    file_name: str,
) -> dict:
    """
    Loads and processes a CSV file containing DSM unit data, where each unit may consist of multiple components
    (technologies) under the same plant name. The function groups data by plant name, processes each group to
    handle different technologies, and organizes the data into a structured DataFrame. It then splits the DataFrame
    based on unique unit_types.

    Args:
        path (str): The directory path where the CSV file is located.
        config (dict): Configuration dictionary, potentially used for specifying additional options or behaviors
                       (not used in the current implementation but provides flexibility for future enhancements).
        file_name (str): The name of the CSV file to be loaded.

    Returns:
        dict: A dictionary where each key is a unique unit_type and the value is a DataFrame containing
              the corresponding DSM units of that type.

    Note:
        - The CSV file is expected to have columns such as 'name', 'technology', 'unit_type', and other operational parameters.
        - The function assumes that the first non-null value in common and bidding columns is representative if multiple
          entries exist for the same plant.
        - It is crucial that the input CSV file follows the expected structure for the function to process it correctly.
    """

    # Load the DSM units file
    # Note: check_duplicates is set to False to avoid raising an error for duplicate unit names
    dsm_units = load_file(
        path=path,
        config=config,
        file_name=file_name,
        check_duplicates=False,
    )

    if dsm_units is None:
        return None

    # Define columns that are common across different technologies within the same plant
    common_columns = [
        "unit_operator",
        "objective",
        "demand",
        "cost_tolerance",
        "unit_type",
        "node",
        "flexibility_measure",
        "is_prosumer",
        "congestion_threshold",
        "peak_load_cap",
    ]
    # Filter the common columns to only include those that exist in the DataFrame
    common_columns = [col for col in common_columns if col in dsm_units.columns]

    # Get bidding columns dynamically
    bidding_columns = [col for col in dsm_units.columns if col.startswith("bidding_")]

    # Initialize the dictionary to hold the final structured data
    dsm_units_dict = {}

    # Process each group of components by plant name or building name
    for name, group in dsm_units.groupby(dsm_units.index):
        dsm_unit = {}

        # Aggregate or select appropriate data for available common and bidding columns
        for col in common_columns + bidding_columns:
            non_null_values = group[col].dropna()
            if not non_null_values.empty:
                dsm_unit[col] = non_null_values.iloc[0]

        # Process each technology within the plant
        components = {}
        for tech, tech_data in group.groupby("technology"):
            # Clean the technology-specific data: drop all-NaN columns and drop 'technology', common, and bidding columns
            cleaned_data = tech_data.dropna(axis=1, how="all").drop(
                columns=["technology"] + common_columns + bidding_columns,
                errors="ignore",
            )
            # Ensure that there is at least one record before adding to components
            if not cleaned_data.empty:
                components[tech] = cleaned_data.to_dict(orient="records")[0]

        dsm_unit["components"] = components
        dsm_units_dict[name] = dsm_unit

    # Convert the structured dictionary into a DataFrame
    dsm_units_df = pd.DataFrame.from_dict(dsm_units_dict, orient="index")

    # Split the DataFrame based on unit_type
    unit_type_dict = {}
    if "unit_type" in dsm_units_df.columns:
        for unit_type in dsm_units_df["unit_type"].unique():
            unit_type_dict[unit_type] = dsm_units_df[
                dsm_units_df["unit_type"] == unit_type
            ]

    return unit_type_dict


def replace_paths(config: dict, inputs_path: str):
    """
    This function replaces all config items which end with "_path"
    to one starting with the given inputs_path.
    So that paths in the config are relative to the inputs_path where the config is read from.

    Args:
        config (dict): the config dict read from yaml
        inputs_path (str): the base path from the config

    Returns:
        dict: the adjusted config dict
    """

    if isinstance(config, dict):
        for key, value in config.items():
            if isinstance(value, dict | list):
                config[key] = replace_paths(value, inputs_path)
            elif isinstance(key, str) and key.endswith("_path") and value is not None:
                # Skip values that are already absolute filesystem paths — this
                # is critical for paired-scenario staggered training, where the
                # shared trained-policies path is set programmatically (and
                # absolute) on a per-world scenario_data after each world's
                # primary inputs_path has already been resolved.
                if os.path.isabs(value):
                    continue
                if not value.startswith(inputs_path):
                    config[key] = inputs_path + "/" + value
    elif isinstance(config, list):
        for i, item in enumerate(config):
            config[i] = replace_paths(item, inputs_path)
    return config


def make_market_config(
    id: str,
    market_params: dict,
    world_start: datetime,
    world_end: datetime,
) -> MarketConfig:
    """
    Create a market config from a given dictionary.

    Args:
    id (str): The id of the market.
    market_params (dict): The market parameters.
    world_start (datetime.datetime): The start time of the world.
    world_end (datetime.datetime): The end time of the world.

    Returns:
    MarketConfig: The market config.
    """
    freq, interval = convert_to_rrule_freq(market_params["opening_frequency"])
    start = market_params.get("start_date")
    end = market_params.get("end_date")
    if start:
        start = pd.Timestamp(start)
    if end:
        end = pd.Timestamp(end)
    start = start or world_start
    end = end or world_end

    market_products = [
        MarketProduct(
            duration=pd.Timedelta(product["duration"]),
            count=product["count"],
            first_delivery=pd.Timedelta(product["first_delivery"]),
        )
        for product in market_params["products"]
    ]
    market_config = MarketConfig(
        market_id=id,
        market_products=market_products,
        product_type=market_params.get("product_type", "energy"),
        opening_hours=rr.rrule(
            freq=freq,
            interval=interval,
            dtstart=start,
            until=end,
            cache=True,
        ),
        opening_duration=pd.Timedelta(market_params["opening_duration"]),
        market_mechanism=market_params["market_mechanism"],
        maximum_bid_volume=market_params.get("maximum_bid_volume", 1e6),
        maximum_bid_price=market_params.get("maximum_bid_price", 3000),
        minimum_bid_price=market_params.get("minimum_bid_price", -3000),
        maximum_gradient=market_params.get("max_gradient"),
        volume_unit=market_params.get("volume_unit", "MW"),
        volume_tick=market_params.get("volume_tick"),
        price_unit=market_params.get("price_unit", "€/MWh"),
        price_tick=market_params.get("price_tick"),
        additional_fields=market_params.get("additional_fields", []),
        supports_get_unmatched=market_params.get("supports_get_unmatched", False),
        param_dict=market_params.get("param_dict", {}),
    )

    return market_config


def read_grid(
    network_path: str | Path, storage_units: pd.DataFrame | None = None
) -> dict[str, pd.DataFrame | None]:
    network_path = Path(network_path)
    buses = None
    lines = None
    generators = None
    loads = None

    if (network_path / "buses.csv").exists():
        buses = pd.read_csv(network_path / "buses.csv", index_col=0)
    if (network_path / "lines.csv").exists():
        lines = pd.read_csv(network_path / "lines.csv", index_col=0)
    if (network_path / "powerplant_units.csv").exists():
        generators = pd.read_csv(network_path / "powerplant_units.csv", index_col=0)
    if (network_path / "demand_units.csv").exists():
        loads = pd.read_csv(network_path / "demand_units.csv", index_col=0)
    if storage_units is None and (network_path / "storage_units.csv").exists():
        storage_units = pd.read_csv(network_path / "storage_units.csv", index_col=0)

    return {
        "buses": buses,
        "lines": lines,
        "generators": generators,
        "loads": loads,
        "storage_units": storage_units,
    }


def add_units(
    units_df: pd.DataFrame,
    unit_type: str,
    world: World,
    forecaster: UnitForecaster,
) -> None:
    """
    Add units to the world from a given dataframe.
    The callback is used to adjust unit_params depending on the unit_type, before adding the unit to the world.

    Args:
        units_df (pandas.DataFrame): The dataframe containing the units.
        unit_type (str): The type of the unit.
        world (World): The world to which the units will be added.
        forecaster (Forecaster): The forecaster used for adding the units.
    """
    if units_df is None:
        return

    logger.info(f"Adding {unit_type} units")

    units_df = units_df.fillna(0)
    for unit_name, unit_params in units_df.iterrows():
        bidding_strategies = bidding_strategies_from_param_dict(unit_params)
        unit_params["bidding_strategies"] = bidding_strategies
        operator_id = unit_params["unit_operator"]
        del unit_params["unit_operator"]
        world.add_unit(
            id=unit_name,
            unit_type=unit_type,
            unit_operator_id=operator_id,
            unit_params=unit_params,
            forecaster=forecaster,
        )


def read_units(
    units_df: pd.DataFrame,
    unit_type: str,
    forecaster: dict[str, UnitForecaster],
    world_bidding_strategies: dict[str, BaseStrategy],
    learning_mode: bool = False,
) -> dict[str, list[dict]]:
    """
    Read units from a dataframe and only add them to a dictionary.
    The dictionary contains the operator ids as keys and the list of units belonging to the operator as values.

    Args:
        units_df (pandas.DataFrame): The dataframe containing the units.
        unit_type (str): The type of the unit.
        forecaster (Forecaster): The forecaster used for adding the units.
        world_bidding_strategies (dict[str, BaseStrategy]): The strategies available in the world
        learning_mode (bool, optional): Whether the world is in learning mode. Defaults to False.
    """
    if units_df is None:
        return {}

    logger.info(f"Adding {unit_type} units")
    units_dict = defaultdict(list)

    units_df = units_df.fillna(0)
    for unit_name, unit_params in units_df.iterrows():
        bidding_strategies = {
            key.split("bidding_")[1]: unit_params[key]
            for key in unit_params.keys()
            if key.startswith("bidding_") and unit_params[key]
        }
        unit_params["bidding_strategies"] = bidding_strategies

        # adjust the unit operator to Operator-RL if learning mode is enabled
        if learning_mode:
            operator_id = adjust_unit_operator_for_learning(
                bidding_strategies,
                world_bidding_strategies,
                unit_params["unit_operator"],
            )
        else:
            operator_id = unit_params["unit_operator"]

        del unit_params["unit_operator"]
        units_dict[operator_id].append(
            dict(
                id=unit_name,
                unit_type=unit_type,
                unit_operator_id=operator_id,
                unit_params=unit_params.to_dict(),
                forecaster=forecaster[unit_name],
            )
        )
    return units_dict


def load_srmc_congestion_from_db(
    db_uri: str, simulation_id: str, index: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    Load a frozen SRMC congestion forecast from the ``grid_flows`` database table.

    Queries the ``grid_flows`` table for the given *simulation_id*, extracts the
    ``congestion_pct`` column, pivots to wide format, and renames each column to
    ``congestion_{line_id}`` so it is consumed by the
    ``congestion_signal_lines_load_from_df`` preprocess algorithm.

    The result is reindexed to *index* using forward-fill to handle any gaps between
    the SRMC run and the learning-run horizon.

    Args:
        db_uri:        SQLAlchemy-compatible DB URI (e.g. ``"sqlite:///local_db/my.db"``).
        simulation_id: The ``simulation`` column value written by the SRMC run.
        index:         The DatetimeIndex of the current scenario.

    Returns:
        DataFrame with columns ``congestion_{line_id}`` indexed by datetime,
        or an empty DataFrame (with the correct *index*) when the table/column is
        missing or the simulation is not found.
    """
    from sqlalchemy import create_engine, inspect, text

    engine = create_engine(db_uri)
    try:
        with engine.connect() as conn:
            inspector = inspect(engine)
            if "grid_flows" not in inspector.get_table_names():
                logger.warning(
                    "Table 'grid_flows' not found in DB '%s'. "
                    "Returning empty SRMC congestion forecast.",
                    db_uri,
                )
                return pd.DataFrame(index=index)

            cols = [c["name"] for c in inspector.get_columns("grid_flows")]
            if "congestion_pct" not in cols:
                logger.warning(
                    "'grid_flows' table in '%s' has no 'congestion_pct' column. "
                    "Run an SRMC simulation with log_flows: true first.",
                    db_uri,
                )
                return pd.DataFrame(index=index)

            df = pd.read_sql(
                text(
                    "SELECT datetime, line, congestion_pct FROM grid_flows"
                    " WHERE simulation = :sim"
                ),
                conn,
                params={"sim": simulation_id},
                parse_dates=["datetime"],
            )
    finally:
        engine.dispose()

    if df.empty:
        logger.warning(
            "No grid_flows rows found for simulation_id='%s'. "
            "Returning empty SRMC congestion forecast.",
            simulation_id,
        )
        return pd.DataFrame(index=index)

    wide = df.pivot_table(
        index="datetime", columns="line", values="congestion_pct", aggfunc="first"
    )
    wide.columns = [f"congestion_{col}" for col in wide.columns]
    wide.index = pd.to_datetime(wide.index)

    # Reindex to scenario horizon; forward-fill gaps, then back-fill leading NaNs
    wide = wide.reindex(index).ffill().bfill()

    logger.info(
        "Loaded SRMC congestion forecast for simulation '%s': %d lines, %d timesteps.",
        simulation_id,
        len(df["line"].unique()),
        len(wide),
    )
    return wide


def save_unique_forecasts(units, save_path: Path) -> None:
    """Collect unique forecasts computed by unit forecasters and write them to CSV.

    Since there is one forecaster per unit but forecasts are shared across units
    (via ``@lru_cache`` on the underlying algorithms), forecasts are deduplicated
    by column name. Column names mirror the ``forecasts_df.csv`` convention so the
    resulting file can be consumed as a drop-in input in a later run.
    """
    unique_forecasts = {
        "price": {},
        "residual_load": {},
        "congestion_signal": {},
        "renewable_utilisation": {},
    }
    default_values = {
        "price": "price_naive_forecast",
        "residual_load": "residual_load_naive_forecast",
        "congestion_signal": "congestion_signal_naive_forecast",
        "renewable_utilisation": "renewable_utilisation_naive_forecast",
    }
    # Track which unit provides per-line congestion signals (deduplicated by algorithm)
    unique_line_congestion_units: dict[str, object] = {}
    default_line_congestion_alg = "congestion_signal_line_naive_forecast"

    for unit in units:
        algs = unit.forecaster.forecast_algorithms
        if isinstance(unit.forecaster, DsmUnitForecaster):
            for key in unique_forecasts:
                forecast_name = algs.get(key, default_values[key])
                unique_forecasts[key][forecast_name] = unit
        else:
            for key in ["price", "residual_load"]:
                forecast_name = algs.get(key, default_values[key])
                unique_forecasts[key][forecast_name] = unit

        # All unit types: track per-line congestion signals when available
        if unit.forecaster.congestion_signal_lines:
            forecast_name = algs.get(
                "congestion_signal_lines", default_line_congestion_alg
            )
            unique_line_congestion_units[forecast_name] = unit

    forecast_dict = {}
    for f_type in unique_forecasts:  # price, residual_load, ...
        for f_name in unique_forecasts[f_type]:  #
            unit = unique_forecasts[f_type][f_name]
            attr_name = (
                "renewable_utilisation_signal"
                if f_type == "renewable_utilisation"
                else f_type
            )
            forecast = getattr(unit.forecaster, attr_name)
            if isinstance(forecast, dict):
                for f_key in forecast:
                    forecast_dict[f"{f_name}_{f_key}"] = forecast[f_key].as_pd_series(
                        name=f"{f_name}_{f_key}"
                    )
            else:
                forecast_dict[f"{f_name}"] = forecast.as_pd_series(name=f"{f_name}")

    # Export per-line congestion signals as congestion_{line_id} columns
    for _f_name, unit in unique_line_congestion_units.items():
        for line_id, series in unit.forecaster.congestion_signal_lines.items():
            col_name = f"congestion_{line_id}"
            forecast_dict[col_name] = series.as_pd_series(name=col_name)

    if not forecast_dict:
        logger.info("No unique forecasts to save.")
        return

    df = pd.concat(forecast_dict.values(), axis=1, names=forecast_dict.keys())
    df.index.name = "datetime"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path)

    logger.info(f"Saved {len(df.columns)} unique forecasts to {save_path}")


def load_config_and_create_forecaster(
    inputs_path: str,
    scenario: str,
    study_case: str,
    extra_units: dict[str, pd.DataFrame] | None = None,
) -> dict[str, object]:
    """
    Load the configuration and files for a given scenario and study case. This function
    allows us to load the files and config only once when running multiple iterations of the same scenario.

    Args:
        inputs_path (str): The path to the folder containing input files necessary for the scenario.
        scenario (str): The name of the scenario to be loaded.
        study_case (str): The specific study case within the scenario to be loaded.
        extra_units (dict[str, pd.DataFrame] | None): Optional foreign-unit DataFrames
            to merge into this scenario for staggered (paired) training. Keys are
            ``"powerplant_units"``, ``"storage_units"``, and ``"demand_units"``;
            values are DataFrames whose rows describe units that natively belong
            to the *other* paired scenario. Foreign units are appended to the local
            unit DataFrames and forced to ``availability = 0`` over the entire
            simulation horizon (with demand also forced to ``0``), so the local
            scenario's grid and clearing remain unchanged. See the D3 staggered
            training spec for details.

    Returns:
        dict[str, object]:: A dictionary containing the configuration and loaded files for the scenario and study case.
    """

    path = f"{inputs_path}/{scenario}"
    logger.info(f"Input files path: {path}")
    logger.info(f"Study case: {study_case}")

    with open(f"{path}/config.yaml") as f:
        config = yaml.safe_load(f)
    if not study_case:
        study_case = list(config.keys())[0]
    config = config[study_case]

    # Set seed, or disable with `seed: null` in config
    set_random_seed(config.get("seed", 42))

    simulation_id = config.get("simulation_id", f"{scenario}_{study_case}")

    logger.info(f"Simulation ID: {simulation_id}")

    start = pd.Timestamp(config["start_date"])
    end = pd.Timestamp(config["end_date"])

    index = pd.date_range(
        start=start,
        end=end,
        freq=config["time_step"],
    )

    unit_operators = load_file(path=path, config=config, file_name="unit_operators")
    powerplant_units = load_file(path=path, config=config, file_name="powerplant_units")
    storage_units = load_file(path=path, config=config, file_name="storage_units")
    demand_units = load_file(path=path, config=config, file_name="demand_units")
    exchange_units = load_file(path=path, config=config, file_name="exchange_units")

    # Merge foreign-unit rows for staggered training before any downstream processing.
    # Tracks foreign ids so we can later force availability=0 / demand=0.
    foreign_unit_ids: dict[str, list[str]] = {
        "powerplant_units": [],
        "storage_units": [],
        "demand_units": [],
    }
    if extra_units:
        for ut_name in ("powerplant_units", "storage_units", "demand_units"):
            extra_df = extra_units.get(ut_name)
            if extra_df is None or extra_df.empty:
                continue
            local_df = {
                "powerplant_units": powerplant_units,
                "storage_units": storage_units,
                "demand_units": demand_units,
            }[ut_name]
            if local_df is None:
                merged = extra_df.copy()
            else:
                # only append rows whose ids aren't already present locally
                new_ids = [uid for uid in extra_df.index if uid not in local_df.index]
                if not new_ids:
                    continue
                merged = pd.concat([local_df, extra_df.loc[new_ids]])
                foreign_unit_ids[ut_name].extend(new_ids)
            if ut_name == "powerplant_units":
                powerplant_units = merged
            elif ut_name == "storage_units":
                storage_units = merged
            else:
                demand_units = merged
    else:
        # No paired training. Honor an optional ``foreign_units.json`` manifest
        # (written next to a generated ``_superset`` folder) so the superset
        # scenario can be run standalone in non-learning mode with the learned
        # policies while foreign generators are forced to zero output. The union
        # unit rows are already present in this folder's CSVs; we only record
        # which ids are foreign so they can be neutralised below.
        manifest_path = Path(path) / "foreign_units.json"
        if manifest_path.exists():
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
            manifest_foreign = manifest.get("foreign_unit_ids", {}) or {}
            local_dfs = {
                "powerplant_units": powerplant_units,
                "storage_units": storage_units,
                "demand_units": demand_units,
            }
            for ut_name in ("powerplant_units", "storage_units", "demand_units"):
                local_df = local_dfs[ut_name]
                local_index = (
                    set(local_df.index.astype(str)) if local_df is not None else set()
                )
                for uid in manifest_foreign.get(ut_name, []) or []:
                    uid = str(uid)
                    if uid not in local_index:
                        raise ValueError(
                            f"foreign_units.json lists foreign unit '{uid}' of type "
                            f"'{ut_name}', but it was not found among the loaded "
                            f"{ut_name} in '{path}'. The superset folder is out of "
                            "sync with its manifest; regenerate it."
                        )
                    foreign_unit_ids[ut_name].append(uid)

    if powerplant_units is None or demand_units is None:
        raise ValueError("No power plant or no demand units were provided!")

    if ((demand_units["min_power"] < 0) & (demand_units["max_power"] > 0)).any() or (
        (demand_units["min_power"] > 0) & (demand_units["max_power"] < 0)
    ).any():
        raise ValueError(
            "min_power and max_power must both be either negative or positive"
        )
    demand_units["min_power"] = -abs(demand_units["min_power"])
    demand_units["max_power"] = -abs(demand_units["max_power"])

    if storage_units is not None:
        if "max_power_charge" in storage_units.columns:
            storage_units["max_power_charge"] = -abs(storage_units["max_power_charge"])
        if "min_power_charge" in storage_units.columns:
            storage_units["min_power_charge"] = -abs(storage_units["min_power_charge"])
        if "capacity" not in storage_units.columns:
            raise ValueError("No capacity column provided for storage units!")

    # Initialize an empty dictionary to combine the DSM units
    dsm_units = {}
    for unit_type in ["industrial_dsm_units", "residential_dsm_units"]:
        units = load_dsm_units(
            path=path,
            config=config,
            file_name=unit_type,
        )
        if units is not None:
            dsm_units.update(units)

    forecasts_df = load_file(
        path=path, config=config, file_name="forecasts_df", index=index
    )

    # If an SRMC pre-run simulation is specified, load its congestion_pct signals
    # from the DB and merge them as congestion_{line_id} columns into forecasts_df.
    srmc_sim_id = config.get("srmc_congestion_simulation_id")
    if srmc_sim_id:
        db_uri = config.get("db_uri")
        if not db_uri:
            logger.warning(
                "srmc_congestion_simulation_id is set but no db_uri found in config. "
                "Skipping SRMC congestion forecast loading."
            )
        else:
            srmc_df = load_srmc_congestion_from_db(db_uri, srmc_sim_id, index)
            if not srmc_df.empty:
                if forecasts_df is None:
                    forecasts_df = srmc_df
                else:
                    forecasts_df = forecasts_df.join(srmc_df, how="outer")

    demand_df = load_file(path=path, config=config, file_name="demand_df", index=index)
    if demand_df is None:
        # no demand timeseries exist, all demand is elastic. Fill missing demand timeseries with zeros and raise a warning.
        logger.warning(
            "!! No demand_df timeseries provided !! Filling demand_df with zeros. Make sure this is what you actually want."
        )
        demand_df = pd.DataFrame(index=index, columns=demand_units.index, data=0.0)
    elif not demand_df.columns.equals(demand_units.index):
        # there exist demand timeseries, but not for all demand units. Some demand is elastic, some is not. Fill missing demand timeseries with zeros and raise a warning.
        logger.warning(
            "!! Incomplete demand_df timeseries provided !! Filling demand_df for some units with zeros. Make sure this is what you actually want."
        )
        missing_columns = demand_units.index.difference(demand_df.columns)
        for col in missing_columns:
            demand_df[col] = 0.0

    exchanges_df = load_file(
        path=path, config=config, file_name="exchanges_df", index=index
    )
    availability = load_file(
        path=path, config=config, file_name="availability_df", index=index
    )
    # check if availability contains any values larger than 1 and raise a warning
    if availability is not None and availability.max().max() > 1:
        # warn the user that the availability contains values larger than 1
        # and normalize the availability
        logger.warning(
            "Availability contains values larger than 1. This is not allowed. "
            "The availability will be normalized automatically. "
            "The quality of the automatic normalization is not guaranteed."
        )
        availability = normalize_availability(powerplant_units, availability)

    if availability is None:
        availability = pd.DataFrame(index=index)

    # Force foreign-scenario units to availability=0 and (for demand) demand=0
    # so they remain registered/dispatchable but contribute nothing to clearing.
    # ``foreign_unit_ids`` is populated either from paired-training ``extra_units``
    # or from a ``foreign_units.json`` manifest in a standalone ``_superset`` run.
    all_foreign_ids: list[str] = (
        foreign_unit_ids["powerplant_units"]
        + foreign_unit_ids["storage_units"]
        + foreign_unit_ids["demand_units"]
    )
    if all_foreign_ids:
        for uid in all_foreign_ids:
            availability[uid] = 0.0
        for uid in foreign_unit_ids["demand_units"]:
            demand_df[uid] = 0.0

    fuel_prices_df = load_file(
        path=path, config=config, file_name="fuel_prices_df", index=index
    )
    if fuel_prices_df is None:
        fuel_prices_df = pd.DataFrame(index=index)

    if len(fuel_prices_df) <= 1:  # single value provided, extend to full index
        fuel_prices_df.index = index[:1]
        fuel_prices_df = fuel_prices_df.reindex(index, method="ffill")

    forecast_algorithms = config.get("forecast_algorithms", {})

    # create shared unit index for caching!
    shared_unit_index = FastIndex(
        start=index[0], end=index[-1], freq=pd.infer_freq(index)
    )
    unit_forecasts: dict[str, UnitForecaster] = {}
    if powerplant_units is not None:
        for id, plant in powerplant_units.iterrows():
            unit_forecasts[id] = PowerplantForecaster(
                index=shared_unit_index,
                availability=availability.get(id, pd.Series(1.0, index, name=id)),
                fuel_prices=fuel_prices_df,
                forecast_algorithms=get_unit_forecast_algorithms(
                    forecast_algorithms, plant
                ),
            )
    if demand_units is not None:
        for id, demand in demand_units.iterrows():
            unit_forecasts[id] = DemandForecaster(
                index=shared_unit_index,
                availability=availability.get(id, pd.Series(1.0, index, name=id)),
                demand=-demand_df[id].abs(),
                forecast_algorithms=get_unit_forecast_algorithms(
                    forecast_algorithms, demand
                ),
            )
    if storage_units is not None:
        for id, storage in storage_units.iterrows():
            unit_forecasts[id] = UnitForecaster(
                index=shared_unit_index,
                availability=availability.get(id, pd.Series(1.0, index, name=id)),
                forecast_algorithms=get_unit_forecast_algorithms(
                    forecast_algorithms, storage
                ),
            )
    if exchange_units is not None:
        for id, exchange in exchange_units.iterrows():
            unit_forecasts[id] = ExchangeForecaster(
                index=shared_unit_index,
                availability=availability.get(id, pd.Series(1.0, index, name=id)),
                forecast_algorithms=get_unit_forecast_algorithms(
                    forecast_algorithms, exchange
                ),
                volume_export=exchanges_df[f"{id}_export"],
                volume_import=exchanges_df[f"{id}_import"],
            )
    if dsm_units is not None:
        for type, dsm in dsm_units.items():
            for id, unit in dsm.iterrows():
                unit_forecast_algorithms = get_unit_forecast_algorithms(
                    forecast_algorithms, unit
                )
                if type == "building":

                    def get_building_profile(column_name: str) -> pd.Series:
                        default_profile = pd.Series(0.0, index=index, name=column_name)
                        if forecasts_df is None:
                            return default_profile
                        return forecasts_df.get(column_name, default_profile)

                    # Base aggregate building profiles
                    building_load_profile = get_building_profile(f"{id}_load_profile")
                    building_heat_demand = get_building_profile(f"{id}_heat_demand")
                    building_pv_profile = get_building_profile(f"{id}_pv_profile")
                    building_battery_profile = get_building_profile(
                        f"{id}_battery_load_profile"
                    )
                    building_ev_profile = get_building_profile(f"{id}_ev_load_profile")
                    building_electricity_price_flex = get_building_profile(
                        f"{id}_electricity_price_flex"
                    )

                    # collect arbitrary component-level forecasts for this building
                    extra_building_profiles = {}

                    if forecasts_df is not None:
                        building_prefix = f"{id}_"
                        for col in forecasts_df.columns:
                            if col.startswith(building_prefix):
                                extra_building_profiles[col] = forecasts_df[col]

                    unit_forecasts[id] = BuildingForecaster(
                        index=shared_unit_index,
                        availability=availability.get(
                            id, pd.Series(1.0, index, name=id)
                        ),
                        forecast_algorithms=unit_forecast_algorithms,
                        fuel_prices=fuel_prices_df,
                        load_profile=building_load_profile,
                        ev_load_profile=building_ev_profile,
                        heat_demand=building_heat_demand,
                        battery_load_profile=building_battery_profile,
                        pv_profile=building_pv_profile,
                        electricity_price_flex=building_electricity_price_flex,
                        **extra_building_profiles,
                    )
                if type == "steel_plant":
                    unit_forecasts[id] = SteelplantForecaster(
                        index=shared_unit_index,
                        availability=availability.get(
                            id, pd.Series(1.0, index, name=id)
                        ),
                        forecast_algorithms=unit_forecast_algorithms,
                        fuel_prices=fuel_prices_df,
                    )
                if type == "hydrogen_plant":
                    unit_forecasts[id] = HydrogenForecaster(
                        index=shared_unit_index,
                        availability=availability.get(
                            id, pd.Series(1.0, index, name=id)
                        ),
                        forecast_algorithms=unit_forecast_algorithms,
                        hydrogen_demand=unit["demand"],
                        seasonal_storage_schedule=0,  # TODO
                    )
                if type == "steam_plant":
                    unit_forecasts[id] = SteamgenerationForecaster(
                        index=shared_unit_index,
                        availability=availability.get(
                            id, pd.Series(1.0, index, name=id)
                        ),
                        forecast_algorithms=unit_forecast_algorithms,
                        demand=unit["demand"],
                        fuel_prices=fuel_prices_df,
                        electricity_price_flex=0,  # TODO
                        thermal_storage_schedule=0,  # TODO
                        thermal_demand=0,  # TODO
                    )
    # Mark foreign units so their (forced-off) transitions can be masked out of the
    # shared MATD3 gradient without conflating them with native units that merely have
    # zero availability this period (e.g. solar at night).
    for uid in all_foreign_ids:
        if uid in unit_forecasts:
            unit_forecasts[uid].is_foreign = True

    price_forecast_source = config.get("price_forecast_source", "auto")
    for unit_forecaster in unit_forecasts.values():
        unit_forecaster.price_forecast_source = price_forecast_source

    return {
        "config": config,
        "simulation_id": simulation_id,
        "path": path,
        "start": start,
        "end": end,
        "unit_operators": unit_operators,
        "powerplant_units": powerplant_units,
        "storage_units": storage_units,
        "demand_units": demand_units,
        "exchange_units": exchange_units,
        "dsm_units": dsm_units,
        "unit_forecasts": unit_forecasts,
        "index": index,
        "forecasts_df": forecasts_df,
    }


def setup_world(
    world: World,
    evaluation_mode: bool = False,
    terminate_learning: bool = False,
    episode: int = 1,
    eval_episode: int = 1,
) -> None:
    """
    Load a scenario from a given path.

    This function loads a scenario within a specified study case from a given path, setting up the world environment for simulation and learning.

    Args:
        world (World): An instance of the World class representing the simulation environment.
        evaluation_mode (bool, optional): A flag indicating whether evaluation should be performed. Defaults to False.
        terminate_learning (bool, optional): An automatically set flag indicating that we terminated the learning process now, either because we reach the end of the episode iteration or because we triggered an early stopping.
        episode (int, optional): The episode number for learning. Defaults to 1.
        eval_episode (int, optional): The episode number for evaluation. Defaults to 1.

    Raises:
        ValueError: If the specified scenario or study case is not found in the provided inputs.

    """
    # make a deep copy of the scenario data to avoid changing the original data
    scenario_data = copy.deepcopy(world.scenario_data)

    simulation_id = scenario_data["simulation_id"]
    config = scenario_data["config"]
    start = scenario_data["start"]
    end = scenario_data["end"]
    unit_operators = scenario_data["unit_operators"]
    powerplant_units = scenario_data["powerplant_units"]
    storage_units = scenario_data["storage_units"]
    demand_units = scenario_data["demand_units"]
    exchange_units = scenario_data["exchange_units"]
    dsm_units = scenario_data["dsm_units"]
    unit_forecasts = scenario_data["unit_forecasts"]
    forecasts_df = scenario_data["forecasts_df"]

    # save every thousand steps by default to free up memory
    save_frequency_hours = config.get("save_frequency_hours", 48)
    # if save_frequency_hours is set to 0, disable saving
    save_frequency_hours = None if save_frequency_hours == 0 else save_frequency_hours
    # check that save_frequency_hours is either None or an integer and raise an error if not with a hint for the user
    if save_frequency_hours is not None and (
        not isinstance(save_frequency_hours, int) or save_frequency_hours <= 0
    ):
        raise ValueError(
            f"save_frequency_hours argument in the config file must be either null or a positive integer. "
            f"Current value: {save_frequency_hours}."
        )

    # Disable save frequency if CSV export is enabled
    if world.export_csv_path and save_frequency_hours is not None:
        save_frequency_hours = None
        logger.info(
            "save_frequency_hours is disabled due to CSV export being enabled. "
            "Data will be stored in the CSV files at the end of the simulation."
        )

        # If PostgreSQL database is in use, warn the user about end-of-simulation saving
        if world.db_uri is not None and "postgresql" in world.db_uri:
            logger.warning(
                "Data will be stored in the PostgreSQL database only at the end of the simulation due to CSV export being enabled. "
                "Disable CSV export to save data at regular intervals (export_csv_path = '')."
            )

    bidding_params = config.get("bidding_strategy_params", {})

    if config.get("learning_mode"):
        raise ValueError(
            "The 'learning_mode' parameter in the top-level of the config.yaml has been moved to 'learning_config'. "
            "Please adjust your config file accordingly."
        )

    # handle initial learning parameters before learning_role exists
    learning_dict = config.get("learning_config", {})
    # those settings need to be overridden before passing to the LearningConfig
    if learning_dict:
        # make sure that continue_learning implies learning_mode
        if learning_dict.get("continue_learning"):
            learning_dict["learning_mode"] = True
        # determined by learning loop in run_learning()
        learning_dict["evaluation_mode"] = evaluation_mode

        if terminate_learning:
            learning_dict["learning_mode"] = False
            learning_dict["evaluation_mode"] = False

        # default path for saving trained policies is set here because
        # a) depends on the simulation_id
        # b) it is set relative to inputs_path in replace_paths() below
        if not learning_dict.get("trained_policies_save_path"):
            learning_dict["trained_policies_save_path"] = (
                f"learned_strategies/{simulation_id}"
            )

    # learning mode always needed for reading units below
    learning_mode = learning_dict.get("learning_mode", False)

    # all paths should be relative to the inputs_path
    config = replace_paths(config, scenario_data["path"])

    world.reset()

    world.setup(
        start=start,
        end=end,
        save_frequency_hours=save_frequency_hours,
        simulation_id=simulation_id,
        learning_dict=learning_dict,
        episode=episode,
        eval_episode=eval_episode,
        bidding_params=bidding_params,
        index=scenario_data["index"],
    )

    # get the market config from the config file and add the markets
    logger.info("Adding markets")
    for market_id, market_params in config["markets_config"].items():
        market_config = make_market_config(
            id=market_id,
            market_params=market_params,
            world_start=start,
            world_end=end,
        )
        if "network_path" in market_config.param_dict.keys():
            grid_data = read_grid(
                market_config.param_dict["network_path"], storage_units=storage_units
            )
            market_config.param_dict["grid_data"] = grid_data

        operator_id = str(market_params["operator"])
        if operator_id not in world.market_operators:
            world.add_market_operator(id=operator_id)

        world.add_market(
            market_operator_id=operator_id,
            market_config=market_config,
        )

    # create list of units from dataframes before adding actual operators
    logger.info("Read units from dataframe")

    units = defaultdict(list)
    powerplant_units = read_units(
        units_df=powerplant_units,
        unit_type="power_plant",
        forecaster=unit_forecasts,
        world_bidding_strategies=world.bidding_strategies,
        learning_mode=learning_mode,
    )

    storage_units = read_units(
        units_df=storage_units,
        unit_type="storage",
        forecaster=unit_forecasts,
        world_bidding_strategies=world.bidding_strategies,
        learning_mode=learning_mode,
    )

    demand_units = read_units(
        units_df=demand_units,
        unit_type="demand",
        forecaster=unit_forecasts,
        world_bidding_strategies=world.bidding_strategies,
        learning_mode=learning_mode,
    )

    exchange_units = read_units(
        units_df=exchange_units,
        unit_type="exchange",
        forecaster=unit_forecasts,
        world_bidding_strategies=world.bidding_strategies,
    )

    if dsm_units is not None:
        for unit_type, units_df in dsm_units.items():
            dsm_units = read_units(
                units_df=units_df,
                unit_type=unit_type,
                forecaster=unit_forecasts,
                world_bidding_strategies=world.bidding_strategies,
                learning_mode=learning_mode,
            )
        for op, op_units in dsm_units.items():
            units[op].extend(op_units)

    for op, op_units in powerplant_units.items():
        units[op].extend(op_units)
    for op, op_units in storage_units.items():
        units[op].extend(op_units)
    for op, op_units in demand_units.items():
        units[op].extend(op_units)
    for op, op_units in exchange_units.items():
        units[op].extend(op_units)

    if unit_operators is not None:
        logger.info("Create unit_operators for portfolio strategies")
        unit_operators_strategies = unit_operators.to_dict("index")
        # remove starting "bidding_" string from market names
        for operator in unit_operators_strategies.keys():
            raw_strategies = unit_operators_strategies[operator]
            converted_strategies = bidding_strategies_from_param_dict(raw_strategies)
            unit_operators_strategies[operator] = converted_strategies
    else:
        unit_operators_strategies = {}

    # if distributed_role is true - there is a manager available
    # and we can add each units_operator as a separate process
    if world.distributed_role is True:
        logger.info("Adding unit operators and units - with subprocesses")
        for op, op_units in units.items():
            strategies = unit_operators_strategies.get(op, {})
            world.add_units_with_operator_subprocess(op, op_units, strategies)
    else:
        logger.info("Adding unit operators and units")
        for company_name in set(units.keys()):
            strategies = unit_operators_strategies.get(company_name, {})
            world.add_unit_operator(id=str(company_name), strategies=strategies)

        # add the units to corresponding unit operators
        for op, op_units in units.items():
            for unit in op_units:
                world.add_unit(**unit)

    # When use_forecasts_df is False, the loaded forecasts_df does not
    # supersede algorithmic forecast calculation.
    use_forecasts_df = config.get("use_forecasts_df", True)
    world.init_forecasts(forecasts_df if use_forecasts_df else None)

    if config.get("save_forecasts", False):
        forecast_save_file = Path(scenario_data["path"]) / config.get(
            "forecast_save_file",
            "saved_forecasts.csv",
        )
        save_unique_forecasts(world.units.values(), forecast_save_file)

    if (
        world.learning_mode
        and world.learning_role is not None
        and len(world.learning_role.rl_strats) == 0
    ):
        raise ValueError("No RL units/strategies were provided!")


def load_scenario_folder(
    world: World,
    inputs_path: str,
    scenario: str,
    study_case: str,
):
    """
    Load a scenario from a given path.

    This function loads a scenario within a specified study case from a given path, setting up the world environment for simulation and learning.

    Args:
        world (World): An instance of the World class representing the simulation environment.
        inputs_path (str): The path to the folder containing input files necessary for the scenario.
        scenario (str): The name of the scenario to be loaded.
        study_case (str): The specific study case within the scenario to be loaded.

    Raises:
        ValueError: If the specified scenario or study case is not found in the provided inputs.

    Note:
        - The function sets up the world environment based on the provided inputs and configuration files.
        - The function utilizes the specified inputs to configure the simulation environment, including market parameters, unit operators, and forecasting data.
        - After calling this function, the world environment is prepared for further simulation and analysis.

    """

    world.scenario_data = load_config_and_create_forecaster(
        inputs_path, scenario, study_case
    )

    setup_world(world=world)


def _read_unit_index(path: str, file_name: str) -> set[str]:
    """Read a unit CSV from a scenario path and return the set of unit ids.

    Returns an empty set if the file does not exist (e.g. a scenario without
    storage units).
    """
    fp = Path(path) / f"{file_name}.csv"
    if not fp.exists():
        return set()
    df = pd.read_csv(fp, index_col=0)
    return set(df.index.astype(str))


def _read_unit_csv(path: str, file_name: str) -> pd.DataFrame | None:
    """Read a unit CSV (raw, no config redirect) and return the DataFrame or None."""
    fp = Path(path) / file_name
    if fp.suffix != ".csv":
        fp = fp.with_suffix(".csv")
    if not fp.exists():
        return None
    df = pd.read_csv(
        fp,
        index_col=0,
        encoding="utf-8",
        na_values=["n.a.", "None", "-", "none", "nan"],
    )
    df.index = df.index.astype(str)
    return df


def build_staggered_supersets(
    scenario_paths: list[str],
    unit_file_overrides: dict[str, dict[str, str]] | None = None,
) -> tuple[
    list[set[str]],
    dict[str, dict[str, pd.DataFrame]],
]:
    """
    Build the cross-scenario unit supersets used by D3 staggered training.

    Reads ``powerplant_units.csv``, ``storage_units.csv`` and ``demand_units.csv``
    from each scenario folder, computes the union of unit ids per type, and for
    each scenario derives the set of *foreign* rows that must be merged in so the
    union of registered RL agents is identical across both worlds.

    As a side effect, a self-contained ``_superset`` subfolder is written next to
    each scenario. It contains the union unit CSVs, a copy of every other input
    file (``config.yaml``, profile CSVs, ...), and a ``foreign_units.json``
    manifest naming the foreign unit ids. That folder can be run standalone in
    non-learning mode with the learned policies; the loader reads the manifest
    and forces the foreign units to zero output. See
    :func:`run_staggered_evaluation`.

    Args:
        scenario_paths: Filesystem paths to the scenario folders (each containing
            ``config.yaml`` and the unit CSVs).

    Returns:
        Tuple ``(local_ids_per_scenario, extra_units_per_scenario)``:
            - ``local_ids_per_scenario[i]`` — the set of unit ids that
              natively belong to scenario *i* (across all unit types).
            - ``extra_units_per_scenario[scenario_path]`` — dict with keys
              ``"powerplant_units"``, ``"storage_units"``, ``"demand_units"``
              and DataFrame values containing the rows from the *other*
              scenarios that must be merged in (with availability=0). Only
              unit ids missing from this scenario are returned.
    """
    UNIT_TYPES = ("powerplant_units", "storage_units", "demand_units")

    unit_file_overrides = unit_file_overrides or {}

    def read_unit_file(scenario_path: str, unit_type: str) -> pd.DataFrame | None:
        file_name = unit_file_overrides.get(scenario_path, {}).get(unit_type, unit_type)
        return _read_unit_csv(scenario_path, file_name)

    # canonical row per unit id per type, taken from the scenario where it natively lives
    canonical: dict[str, dict[str, pd.Series]] = {ut: {} for ut in UNIT_TYPES}
    local_ids: list[set[str]] = []
    for sp in scenario_paths:
        ids_here: set[str] = set()
        for ut in UNIT_TYPES:
            df = read_unit_file(sp, ut)
            if df is None:
                continue
            for uid, row in df.iterrows():
                ids_here.add(str(uid))
                canonical[ut].setdefault(str(uid), row)
        local_ids.append(ids_here)

    extra_units_per_scenario: dict[str, dict[str, pd.DataFrame]] = {}
    for i, sp in enumerate(scenario_paths):
        local_dfs = {ut: read_unit_file(sp, ut) for ut in UNIT_TYPES}
        extras: dict[str, pd.DataFrame] = {}
        for ut in UNIT_TYPES:
            local_set = (
                set(local_dfs[ut].index.astype(str))
                if local_dfs[ut] is not None
                else set()
            )
            foreign_ids = [uid for uid in canonical[ut] if uid not in local_set]
            if not foreign_ids:
                extras[ut] = pd.DataFrame()
                continue
            extras[ut] = pd.DataFrame(
                [canonical[ut][uid] for uid in foreign_ids],
                index=foreign_ids,
            )
        extra_units_per_scenario[sp] = extras

        # Materialise a self-contained, directly-runnable ``_superset`` scenario
        # folder alongside each scenario. It contains the union unit CSVs plus a
        # ``foreign_units.json`` manifest naming the foreign ids, and a copy of
        # every other input file (config.yaml, profiles, ...) so the folder can be
        # run in non-learning mode with the learned policies via the standard
        # loader. The loader reads the manifest and forces the foreign units to
        # zero output (availability=0 / demand=0) — see
        # ``load_config_and_create_forecaster``.
        try:
            superset_dir = Path(sp) / "_superset"
            superset_dir.mkdir(exist_ok=True)

            unit_csv_names = {f"{ut}.csv" for ut in UNIT_TYPES}

            # 1) Write the union unit CSVs.
            for ut in UNIT_TYPES:
                local_df = local_dfs[ut]
                merged = (
                    pd.concat([local_df, extras[ut]])
                    if (local_df is not None and not extras[ut].empty)
                    else (
                        local_df
                        if local_df is not None
                        else (extras[ut] if not extras[ut].empty else None)
                    )
                )
                if merged is not None:
                    merged.to_csv(superset_dir / f"{ut}.csv")

            # 2) Copy every other input file (config.yaml, profile CSVs, license
            #    sidecars, ...) so the folder is self-contained. The union unit
            #    CSVs written above are not overwritten.
            for entry in Path(sp).iterdir():
                if entry.is_dir():
                    continue
                if entry.name in unit_csv_names:
                    continue
                shutil.copy2(entry, superset_dir / entry.name)

            # 3) Write the foreign-unit manifest read back by the loader.
            manifest = {
                "version": 1,
                "scenario": Path(sp).name,
                "foreign_unit_ids": {
                    ut: [str(uid) for uid in extras[ut].index]
                    for ut in UNIT_TYPES
                    if not extras[ut].empty
                },
            }
            with open(superset_dir / "foreign_units.json", "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
        except OSError as e:  # pragma: no cover — non-fatal, inspection-only
            logger.warning(f"Could not write _superset folder to {sp}: {e}")

    return local_ids, extra_units_per_scenario


# Files that are allowed to differ between paired scenarios (they are merged /
# neutralised by the superset mechanism, so a mismatch is expected and benign).
_STAGGERED_UNIT_CSVS = {
    "powerplant_units.csv",
    "storage_units.csv",
    "demand_units.csv",
}


def _check_staggered_input_file_parity(
    scenario_paths: list[str], names: list[str]
) -> None:
    """Warn when paired staggered scenarios have mismatched optional input files.

    Compares the set of ``.csv`` files present in each scenario folder, ignoring
    the unit-definition CSVs (which are intentionally different between scenarios
    and are handled by the superset mechanism) and any ``.license`` sidecars.

    A mismatch usually means a frozen forecast (e.g. ``forecasts_df.csv``) was
    prepared for one scenario but not the other.  Without it the missing scenario
    silently falls back to the naive congestion / price signal.

    Args:
        scenario_paths: Filesystem paths to the two scenario folders.
        names:          Logical scenario names (for log messages).
    """
    csv_sets: list[set[str]] = []
    for sp in scenario_paths:
        p = Path(sp)
        files = {f.name for f in p.glob("*.csv") if f.name not in _STAGGERED_UNIT_CSVS}
        csv_sets.append(files)

    only_in_first = csv_sets[0] - csv_sets[1]
    only_in_second = csv_sets[1] - csv_sets[0]

    if only_in_first or only_in_second:
        logger.warning(
            "Staggered scenario input file mismatch detected — forecasts may differ "
            "between worlds.  Files present in '%s' but missing from '%s': %s.  "
            "Files present in '%s' but missing from '%s': %s.  "
            "Ensure both scenarios have the same optional input files (e.g. "
            "forecasts_df.csv) so that forecast signals are consistent.",
            names[0],
            names[1],
            sorted(only_in_first) or "none",
            names[1],
            names[0],
            sorted(only_in_second) or "none",
        )


def load_staggered_scenario(
    worlds: list[World],
    inputs_path: str,
    scenario: str,
    study_case: str,
) -> None:
    """
    Load a paired (D3 staggered training) scenario into two ``World`` instances.

    The primary scenario referenced by ``inputs_path/scenario`` must contain a
    ``learning_config.staggered_training`` block listing exactly two scenarios
    (the primary plus one paired scenario). Both scenarios are loaded with the
    cross-scenario unit superset merged in, so each world registers the union
    of RL agents and foreign units sit at availability=0 (G2 / G4 of the spec).

    Each world's ``simulation_id`` is set to its scenario folder name so DB
    outputs are namespaced cleanly (e.g. ``staggered_bau`` vs
    ``staggered_inv``). The configured logical alias is preserved on
    ``world.scenario_data["staggered_scenario_name"]`` for metric reporting.

    Args:
        worlds: Exactly two ``World`` instances; populated in place.
        inputs_path: Inputs root (same as ``load_scenario_folder``).
        scenario: Primary scenario folder name.
        study_case: Study case key inside the primary ``config.yaml``.
    """
    if len(worlds) != 2:
        raise ValueError(
            f"load_staggered_scenario requires exactly 2 World instances, got {len(worlds)}"
        )

    # Read the primary config to discover the paired scenarios.
    primary_path = f"{inputs_path}/{scenario}"
    with open(f"{primary_path}/config.yaml") as f:
        primary_config = yaml.safe_load(f)
    if not study_case:
        study_case = list(primary_config.keys())[0]
    primary_config = primary_config[study_case]
    learning_config = primary_config.get("learning_config", {}) or {}
    staggered = learning_config.get("staggered_training", {}) or {}
    if not staggered.get("enabled"):
        raise ValueError(
            "load_staggered_scenario called but 'learning_config.staggered_training.enabled' is not true."
        )
    scenarios = staggered.get("scenarios", [])
    if len(scenarios) != 2:
        raise ValueError(
            f"learning_config.staggered_training.scenarios must list exactly 2 scenarios, got {len(scenarios)}"
        )
    names = [s.get("name") for s in scenarios]
    if any(not n for n in names) or len(set(names)) != 2:
        raise ValueError(
            f"staggered_training.scenarios must have unique non-empty names; got {names}"
        )
    for n in names:
        if not str(n).replace("_", "").isalnum():
            raise ValueError(
                f"staggered_training scenario name '{n}' must be alphanumeric/underscore (DB-safe)."
            )

    # Resolve scenario paths. Each entry's `path` is interpreted relative to the
    # primary scenario folder so config files can stay portable.
    scenario_paths: list[str] = []
    for s in scenarios:
        p = Path(s["path"])
        if not p.is_absolute():
            p = (Path(primary_path) / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Staggered scenario path does not exist: {p}")
        scenario_paths.append(str(p))

    # Validate that both scenario folders contain the same optional input files.
    # A mismatch (e.g. one scenario has forecasts_df.csv and the other doesn't)
    # typically means a frozen forecast was forgotten, which silently degrades to
    # naive signals for the scenario that is missing the file.
    _check_staggered_input_file_parity(scenario_paths, names)

    # Resolve each active case's unit-file redirects before building the
    # supersets. Otherwise a case-level storage_units override would be loaded
    # into each world after the superset was constructed, breaking G2 parity.
    study_cases_by_path: dict[str, str] = {}
    unit_file_overrides: dict[str, dict[str, str]] = {}
    for sp in scenario_paths:
        with open(Path(sp) / "config.yaml") as f:
            sc_config = yaml.safe_load(f)
        sc_study_case = (
            study_case if study_case in sc_config else list(sc_config.keys())[0]
        )
        study_cases_by_path[sp] = sc_study_case
        sc_case_config = sc_config[sc_study_case]
        unit_file_overrides[sp] = {
            unit_type: str(sc_case_config.get(unit_type, f"{unit_type}.csv"))
            for unit_type in ("powerplant_units", "storage_units", "demand_units")
        }

    # Build the unit supersets across both scenarios.
    _local_ids, extras_by_path = build_staggered_supersets(
        scenario_paths, unit_file_overrides
    )

    # Load each scenario into its world with the foreign rows merged in.
    # Both worlds must share the same asyncio event loop so the orchestrator
    # can advance them in lockstep inside a single ``run_until_complete``.
    for w in worlds[1:]:
        w.loop = worlds[0].loop
    base_sim_id: str | None = None
    rl_unit_id_sets: list[set[str]] = []
    line_id_sets: list[set[str]] = []
    for world, sp, name in zip(worlds, scenario_paths, names):
        sp_path = Path(sp)
        scenario_inputs_path = str(sp_path.parent)
        scenario_name = sp_path.name
        sc_study_case = study_cases_by_path[sp]

        world.scenario_data = load_config_and_create_forecaster(
            scenario_inputs_path,
            scenario_name,
            sc_study_case,
            extra_units=extras_by_path[sp],
        )

        # Namespace simulation_id by the scenario folder name so DB outputs
        # are separable and there is no study-case / scenario-name stutter
        # (e.g. ``staggered_bau`` and ``staggered_inv`` rather than
        # ``staggered_bau_staggered_bau`` / ``staggered_bau_staggered_inv``).
        # The logical scenario alias is preserved separately under
        # ``staggered_scenario_name`` for metric reporting.
        if base_sim_id is None:
            base_sim_id = world.scenario_data["simulation_id"]
        world.scenario_data["simulation_id"] = f"{scenario_name}_{study_case}"
        world.scenario_data["staggered_scenario_name"] = name

        # Setup the world
        setup_world(world=world)

        # Track RL unit ids and grid line ids for cross-world consistency checks.
        rl_unit_id_sets.append(
            set(getattr(world, "learning_role", None).rl_strats.keys())
            if world.learning_role is not None
            else set()
        )
        line_id_sets.append(_collect_market_line_ids(world))

    simulation_ids = [str(w.scenario_data.get("simulation_id", "")) for w in worlds]
    if any(not sid for sid in simulation_ids) or len(set(simulation_ids)) != 2:
        raise ValueError(
            "Staggered training requires two distinct non-empty simulation_id values; "
            f"got {simulation_ids}."
        )

    # G2 — registered RL units must match across both worlds.
    if rl_unit_id_sets[0] != rl_unit_id_sets[1]:
        only_a = rl_unit_id_sets[0] - rl_unit_id_sets[1]
        only_b = rl_unit_id_sets[1] - rl_unit_id_sets[0]
        raise ValueError(
            "Staggered training G2 violation: RL agent ids differ between worlds. "
            f"Only in scenario '{names[0]}': {sorted(only_a)}; "
            f"only in scenario '{names[1]}': {sorted(only_b)}."
        )

    # 5.1 — line ids must be identical (only line capacities may differ).
    if line_id_sets[0] and line_id_sets[1] and line_id_sets[0] != line_id_sets[1]:
        only_a = line_id_sets[0] - line_id_sets[1]
        only_b = line_id_sets[1] - line_id_sets[0]
        raise ValueError(
            "Staggered training requires identical line ids across both scenarios "
            "(only capacities may differ). "
            f"Only in scenario '{names[0]}': {sorted(only_a)}; "
            f"only in scenario '{names[1]}': {sorted(only_b)}."
        )


def _collect_market_line_ids(world: World) -> set[str]:
    """Return the union of line ids across all networked markets registered in *world*.

    Returns an empty set when no market carries a ``grid_data`` payload (e.g.
    scenarios without network constraints).
    """
    line_ids: set[str] = set()
    for market_config in world.markets.values():
        grid_data = (
            market_config.param_dict.get("grid_data")
            if hasattr(market_config, "param_dict")
            else None
        )
        if not grid_data:
            continue
        lines_df = grid_data.get("lines") if isinstance(grid_data, dict) else None
        if lines_df is None:
            continue
        try:
            line_ids.update(str(i) for i in lines_df.index)
        except Exception:  # pragma: no cover — defensive
            continue
    return line_ids


def load_custom_units(
    world: World,
    inputs_path: str,
    scenario: str,
    file_name: str,
    forecast_file_name: str,
    unit_type: str,
) -> None:
    """
    Load custom units from a given path.

    This function loads custom units of a specified type from a given path within a scenario, adding them to the world environment for simulation.

    Args:
        world (World): An instance of the World class representing the simulation environment.
        inputs_path (str): The path to the folder containing input files necessary for the custom units.
        scenario (str): The name of the scenario from which the custom units are to be loaded.
        file_name (str): The name of the file containing the custom units.
        unit_type (str): The type of the custom units to be loaded.

    Example:
        >>> load_custom_units(
            world=world,
            inputs_path="/path/to/inputs",
            scenario="scenario_name",
            file_name="custom_units.csv",
            unit_type="custom_type"
        )

    Note:
        - The function loads custom units from the specified file within the given scenario and adds them to the world environment for simulation.
        - If the specified custom units file is not found, a warning is logged.
        - Each unique unit operator in the custom units is added to the world's unit operators.
        - The custom units are added to the world environment based on their type for use in simulations.
    """
    path = f"{inputs_path}/{scenario}"

    custom_units = load_file(
        path=path,
        config={},
        file_name=file_name,
    )

    if custom_units is None:
        logger.warning(f"No {file_name} units were provided!")

    forecasts = load_file(
        path=path,
        config={},
        file_name=forecast_file_name,
    )
    if forecasts is None:
        logger.warning(f"No {forecast_file_name} forecasts were provided!")

    operators = custom_units.unit_operator.unique()
    for operator in operators:
        if operator not in world.unit_operators:
            world.add_unit_operator(id=str(operator))

    kwargs = {}
    for k, v in forecasts.items():
        kwargs[k] = v
    forecaster = CustomUnitForecaster(forecasts.index, **kwargs)
    add_units(
        units_df=custom_units,
        unit_type=unit_type,
        world=world,
        forecaster=forecaster,
    )


def run_learning(
    world: World,
    verbose: bool = False,
) -> None:
    """
    Train Deep Reinforcement Learning (DRL) agents to act in a simulated market environment.

    This function runs multiple episodes of simulation to train DRL agents, performs evaluation, and saves the best runs. It maintains the buffer and learned agents in memory to avoid resetting them with each new run.

    Args:
        world (World): An instance of the World class representing the simulation environment.
        verbose (bool, optional): A flag indicating whether to enable verbose logging. Defaults to False.

    Note:
        - The function uses a ReplayBuffer to store experiences for training the DRL agents.
        - It iterates through training episodes, updating the agents and evaluating their performance at regular intervals.
        - Initial exploration is active at the beginning and is disabled after a certain number of episodes to improve the performance of DRL algorithms.
        - Upon completion of training, the function performs an evaluation run using the last policy learned during training.
        - The best policies are chosen based on the average reward obtained during the evaluation runs, and they are saved for future use.
    """
    from assume.reinforcement_learning.buffer import ReplayBuffer

    # If staggered (paired-scenario) training is enabled, the single-world
    # training loop below cannot service it — direct the user to the dedicated
    # entry point that builds and orchestrates two ``World`` instances.
    learning_config = (
        world.scenario_data.get("config", {}).get("learning_config", {}) or {}
    )
    staggered = learning_config.get("staggered_training", {}) or {}
    if staggered.get("enabled"):
        raise ValueError(
            "Staggered training is enabled in learning_config. "
            "Use `assume.scenario.loader_csv.run_staggered_learning(...)` "
            "instead of `run_learning(...)` so that two paired worlds are constructed."
        )

    if not verbose:
        # Avoid silencing this module's INFO logs; instead silence very noisy external loggers
        logging.getLogger("mango").setLevel(logging.WARNING)

    # remove csv path so that nothing is written while learning
    temp_csv_path = world.export_csv_path
    world.export_csv_path = ""

    # initialize policies already here to set the obs_dim and act_dim in the learning role
    world.learning_role.rl_algorithm.initialize_policy()

    # check if we already stored policies for this simulation
    save_path = world.learning_role.learning_config.trained_policies_save_path
    continue_learning = world.learning_role.learning_config.continue_learning
    confirm_learning_save_path(save_path, continue_learning)

    # -----------------------------------------
    # Information that needs to be stored across episodes, aka one simulation run
    # Read optional replay-buffer persistence settings from learning_config
    lc = world.learning_role.learning_config
    cfg_save_flag = getattr(lc, "save_replay_buffer", True)
    cfg_save_path = getattr(lc, "replay_buffer_save_path", None)
    cfg_load_flag = getattr(lc, "load_replay_buffer", False)
    cfg_load_path = getattr(lc, "replay_buffer_load_path", None)
    cfg_state_save_flag = getattr(lc, "save_learning_state", True)
    cfg_state_save_path = getattr(lc, "learning_state_save_path", None)
    cfg_state_load_flag = getattr(lc, "load_learning_state", False)
    cfg_state_load_path = getattr(lc, "learning_state_load_path", None)

    # default path next to saved policies
    default_buffer_path = f"{save_path}/last_policies/replay_buffer.npz"
    default_state_path = f"{save_path}/last_policies/learning_state.pt"

    def persist_replay_buffer():
        """Persist the replay buffer to disk alongside saved policies (configurable)."""
        try:
            if cfg_save_flag:
                save_path_cfg = (
                    cfg_save_path if cfg_save_path is not None else default_buffer_path
                )
                # ensure directory exists will be handled by ReplayBuffer.save
                if (
                    hasattr(world.learning_role, "buffer")
                    and world.learning_role.buffer is not None
                ):
                    logger.info(f"Saving replay buffer to {save_path_cfg}")
                    world.learning_role.buffer.save(save_path_cfg)
                    if os.path.exists(save_path_cfg):
                        logger.info(f"Replay buffer saved: {save_path_cfg}")
                    else:
                        logger.warning(
                            f"Replay buffer save attempted but file not found afterwards: {save_path_cfg}"
                        )
        except Exception:
            logger.warning("Failed to save replay buffer")

    def resolve_checkpoint_path(
        load_path: str | None,
        save_path: str | None,
        default_path: str,
    ) -> str:
        """Resolve checkpoint path with precedence: explicit load > explicit save > default."""
        if load_path is not None:
            return load_path
        if save_path is not None:
            return save_path
        return default_path

    # keep tensorboard history when we resume from any persisted state
    resume_mode = continue_learning or cfg_load_flag or cfg_state_load_flag
    if resume_mode:
        logger.info(
            "Resume mode activated (single-world): "
            f"continue_learning={continue_learning}, "
            f"load_replay_buffer={cfg_load_flag}, "
            f"load_learning_state={cfg_state_load_flag}"
        )
    if not resume_mode:
        tensorboard_path = f"tensorboard/{world.scenario_data['simulation_id']}"
        if os.path.exists(tensorboard_path):
            shutil.rmtree(tensorboard_path, ignore_errors=True)

    buffer = None
    # Load only when explicitly requested via load_replay_buffer
    if cfg_load_flag:
        # choose explicit load path if provided, otherwise fall back to configured save path or default
        path_to_load = resolve_checkpoint_path(
            load_path=cfg_load_path,
            save_path=cfg_save_path,
            default_path=default_buffer_path,
        )
        if not os.path.exists(path_to_load):
            raise AssumeException(
                f"load_replay_buffer is true but no buffer file found at {path_to_load}"
            )
        try:
            buffer = ReplayBuffer.load(
                path_to_load,
                device=world.learning_role.device,
                float_type=world.learning_role.float_type,
            )
            logger.info(f"Loaded replay buffer from {path_to_load}")
            # disable initial experience collection when buffer provided
            try:
                world.learning_role.learning_config.episodes_collecting_initial_experience = 0
                logger.info(
                    "Replay buffer provided — skipping initial experience collection (episodes_collecting_initial_experience set to 0)."
                )
            except Exception:
                logger.warning(
                    "Could not set episodes_collecting_initial_experience to 0 on learning_config"
                )
        except Exception as e:
            raise AssumeException(
                f"Failed to load replay buffer from {path_to_load}: {e}"
            )
    else:
        # create fresh buffer
        buffer = ReplayBuffer(
            buffer_size=world.learning_role.learning_config.replay_buffer_size,
            obs_dim=world.learning_role.rl_algorithm.obs_dim,
            act_dim=world.learning_role.rl_algorithm.act_dim,
            n_rl_units=len(world.learning_role.rl_strats),
            device=world.learning_role.device,
            float_type=world.learning_role.float_type,
        )

    inter_episodic_data = {
        "buffer": buffer,
        "actors_and_critics": None,
        "max_eval": defaultdict(lambda: -1e9),
        "all_eval": defaultdict(list),
        "avg_all_eval": [],
        "episodes_done": 0,
        "eval_episodes_done": 0,
    }

    world.learning_role.load_inter_episodic_data(inter_episodic_data)

    if cfg_state_load_flag:
        state_path_to_load = resolve_checkpoint_path(
            load_path=cfg_state_load_path,
            save_path=cfg_state_save_path,
            default_path=default_state_path,
        )
        if not os.path.exists(state_path_to_load):
            raise AssumeException(
                "load_learning_state is true but no learning-state file found at "
                f"{state_path_to_load}"
            )
        try:
            world.learning_role.load_runtime_state(state_path_to_load)
            logger.info(f"Loaded learning runtime state from {state_path_to_load}")
        except Exception as e:
            raise AssumeException(
                f"Failed to load learning runtime state from {state_path_to_load}: {e}"
            )

    inter_episodic_data = world.learning_role.get_inter_episodic_data()
    start_episode = max(int(inter_episodic_data.get("episodes_done", 0)) + 1, 1)
    eval_episode = int(inter_episodic_data.get("eval_episodes_done", 0)) + 1

    validation_interval = world.learning_role.determine_validation_interval()

    # sync train frequency with simulation horizon once at the beginning of training and overwrite scenario data
    world.scenario_data["config"]["learning_config"]["train_freq"] = (
        world.learning_role.sync_train_freq_with_simulation_horizon()
    )

    if start_episode > world.learning_role.learning_config.training_episodes:
        logger.info(
            "Training already complete according to loaded runtime state "
            f"(episodes_done={start_episode - 1}). Skipping training loop."
        )

    if (
        start_episode != 1
        and start_episode <= world.learning_role.learning_config.training_episodes
    ):
        setup_world(
            world=world,
            episode=start_episode,
        )
        world.learning_role.load_inter_episodic_data(inter_episodic_data)

    for episode in tqdm(
        range(
            start_episode,
            world.learning_role.learning_config.training_episodes + 1,
        ),
        desc="Training Episodes",
    ):
        # -----------------------------------------
        # Give the newly initialized learning role the needed information across episodes
        if episode != start_episode:
            setup_world(
                world=world,
                episode=episode,
            )
            world.learning_role.load_inter_episodic_data(inter_episodic_data)

        world.run()

        world.learning_role.tensor_board_logger.update_tensorboard()

        # -----------------------------------------
        # Store updated information across episodes
        inter_episodic_data = world.learning_role.get_inter_episodic_data()
        inter_episodic_data["episodes_done"] = episode

        # persist the replay buffer once the initial experience collection completes
        if (
            episode
            == world.learning_role.learning_config.episodes_collecting_initial_experience
        ):
            persist_replay_buffer()

        # evaluation run:
        if (
            episode % validation_interval == 0
            and episode
            >= world.learning_role.learning_config.episodes_collecting_initial_experience
            + validation_interval
        ):
            world.reset()

            # load evaluation run
            setup_world(
                world=world,
                evaluation_mode=True,
                episode=episode,
                eval_episode=eval_episode,
            )

            world.learning_role.load_inter_episodic_data(inter_episodic_data)

            world.run()

            world.learning_role.tensor_board_logger.update_tensorboard()

            if not world.db_uri:
                raise AssumeException("No learning rewards as no database was given")

            total_rewards = world.output_role.get_sum_reward(episode=eval_episode)

            if len(total_rewards) == 0:
                raise AssumeException("No rewards were collected during evaluation run")

            avg_reward = np.mean(total_rewards)

            # check reward improvement in evaluation run
            # and store best run in eval folder
            terminate = world.learning_role.compare_and_save_policies(
                {"avg_reward": avg_reward}
            )

            inter_episodic_data["eval_episodes_done"] = eval_episode

            # if we have not improved in the last x evaluations, we stop loop
            if terminate:
                break

            eval_episode += 1

        world.reset()

        # save the policies after each episode in case the simulation is stopped or crashes
        if (
            episode
            >= world.learning_role.learning_config.episodes_collecting_initial_experience
            + validation_interval
        ):
            world.learning_role.rl_algorithm.save_params(
                directory=f"{world.learning_role.learning_config.trained_policies_save_path}/last_policies"
            )
            # also persist replay buffer alongside policies (configurable)
            persist_replay_buffer()

            try:
                if cfg_state_save_flag:
                    state_save_path = (
                        cfg_state_save_path
                        if cfg_state_save_path is not None
                        else default_state_path
                    )
                    world.learning_role.save_runtime_state(state_save_path)
                    logger.info(f"Learning runtime state saved: {state_save_path}")
            except Exception:
                logger.warning("Failed to save learning runtime state")

    # container shutdown implicitly with new initialisation
    logger.info("################")
    logger.info("Training finished, Start evaluation run")
    world.export_csv_path = temp_csv_path

    world.reset()

    # latest policies for final simulation run
    world.scenario_data["config"]["learning_config"]["trained_policies_load_path"] = (
        f"{world.learning_role.learning_config.trained_policies_save_path}/last_policies"
    )

    # load scenario for evaluation
    setup_world(
        world=world,
        terminate_learning=True,
    )


def run_staggered_learning(
    inputs_path: str,
    scenario: str,
    study_case: str,
    db_uri: str = "",
    export_csv_path: str = "",
    log_level: str = "INFO",
    verbose: bool = False,
) -> tuple[World, World]:
    """Build two paired ``World`` instances and run D3 staggered MATD3 training.

    The primary scenario folder must declare a
    ``learning_config.staggered_training`` block listing exactly two scenarios
    (the primary plus one paired scenario). Both worlds share the same
    ``db_uri``/``export_csv_path``; their outputs are namespaced by the
    scenario folder name so DB rows from the two worlds can be queried
    separately (e.g. ``staggered_bau`` vs ``staggered_inv``).

    Args:
        inputs_path: Path to the inputs root (containing scenario folders).
        scenario: Primary scenario folder name.
        study_case: Study case key inside the primary scenario's ``config.yaml``.
        db_uri: SQLAlchemy DB URI for both worlds (optional).
        export_csv_path: CSV output path; learning suppresses it during training.
        log_level: Log level forwarded to ``World``.
        verbose: When true, the trainer keeps INFO logging enabled.

    Returns:
        Tuple ``(world_a, world_b)`` for inspection / further teardown by the caller.
    """
    from assume.reinforcement_learning.staggered_trainer import StaggeredTrainer

    world_a = World(
        database_uri=db_uri,
        export_csv_path=export_csv_path,
        log_level=log_level,
    )
    world_b = World(
        database_uri=db_uri,
        export_csv_path=export_csv_path,
        log_level=log_level,
    )

    load_staggered_scenario(
        worlds=[world_a, world_b],
        inputs_path=inputs_path,
        scenario=scenario,
        study_case=study_case,
    )

    learning_config = (
        world_a.scenario_data.get("config", {}).get("learning_config", {}) or {}
    )
    staggered_cfg = learning_config.get("staggered_training", {}) or {}
    swap_order = staggered_cfg.get("swap_order_per_episode", True)

    trainer = StaggeredTrainer(
        worlds=[world_a, world_b],
        swap_order_per_episode=swap_order,
        verbose=verbose,
    )
    trainer.run()
    return world_a, world_b


def run_staggered_evaluation(
    inputs_path: str,
    scenario: str,
    study_case: str,
    trained_policies_path: str | None = None,
    db_uri: str = "",
    export_csv_path: str = "",
    log_level: str = "INFO",
) -> tuple[World, ...]:
    """Run the staggered ``_superset`` scenarios in non-learning mode with learned policies.

    Convenience wrapper around the manifest-driven superset mechanism. It
    (re)builds the cross-scenario ``_superset`` folders (so the union unit CSVs
    and ``foreign_units.json`` manifests are current), then loads and runs each
    superset folder as a standalone, non-learning simulation that loads the
    shared trained policy. Foreign generators (those native to the *other*
    paired scenario) are forced to zero output via the manifest, so each world
    reflects only its own native units while still presenting the full RL agent
    set the shared policy was trained on.

    Args:
        inputs_path: Path to the inputs root (containing scenario folders).
        scenario: Primary scenario folder name (declares the ``staggered_training`` block).
        study_case: Study case key inside the primary scenario's ``config.yaml``.
        trained_policies_path: Directory of saved policies to load (the shared
            policy is the same for every paired world). When ``None``, defaults
            to ``<anchor_scenario>/learned_strategies/<anchor>_<study_case>/last_policies``.
        db_uri: SQLAlchemy DB URI for the worlds (optional).
        export_csv_path: CSV output path (optional).
        log_level: Log level forwarded to ``World``.

    Returns:
        Tuple of the evaluated ``World`` instances (one per paired scenario).
    """
    # Resolve the paired scenarios from the primary config's staggered block.
    primary_path = Path(f"{inputs_path}/{scenario}")
    with open(primary_path / "config.yaml") as f:
        primary_config = yaml.safe_load(f)
    if not study_case:
        study_case = list(primary_config.keys())[0]
    primary_config = primary_config[study_case]
    learning_config = primary_config.get("learning_config", {}) or {}
    staggered = learning_config.get("staggered_training", {}) or {}
    if not staggered.get("enabled"):
        raise ValueError(
            "run_staggered_evaluation called but "
            "'learning_config.staggered_training.enabled' is not true."
        )
    scenarios = staggered.get("scenarios", [])
    if len(scenarios) < 2:
        raise ValueError(
            "learning_config.staggered_training.scenarios must list at least 2 scenarios."
        )

    scenario_paths: list[str] = []
    for s in scenarios:
        p = Path(s["path"])
        if not p.is_absolute():
            p = (primary_path / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Staggered scenario path does not exist: {p}")
        scenario_paths.append(str(p))

    # (Re)build the supersets so the _superset folders + manifests are current.
    build_staggered_supersets(scenario_paths)

    # The shared policy is saved once under the anchor (first) scenario during
    # training; every eval world loads from that same absolute path.
    anchor_path = Path(scenario_paths[0])
    default_policy_path = str(
        anchor_path
        / "learned_strategies"
        / f"{anchor_path.name}_{study_case}"
        / "last_policies"
    )
    policy_path = trained_policies_path or default_policy_path

    worlds: list[World] = []
    for sp in scenario_paths:
        sp_path = Path(sp)

        # Discover the study case inside the copied superset config.
        with open(sp_path / "_superset" / "config.yaml") as f:
            sc_config = yaml.safe_load(f)
        sc_study_case = (
            study_case if study_case in sc_config else list(sc_config.keys())[0]
        )

        world = World(
            database_uri=db_uri,
            export_csv_path=export_csv_path,
            log_level=log_level,
        )
        # Loads the union units and neutralises foreign units via the manifest.
        world.scenario_data = load_config_and_create_forecaster(
            str(sp_path), "_superset", sc_study_case
        )

        # Namespace the simulation id by the scenario folder so DB rows from the
        # paired worlds (and from training) stay separable.
        world.scenario_data["simulation_id"] = f"{sp_path.name}_eval"

        lc = world.scenario_data["config"].setdefault("learning_config", {})
        lc["trained_policies_load_path"] = policy_path

        # Non-learning run with the loaded shared policy.
        setup_world(world=world, terminate_learning=True)
        world.run()
        worlds.append(world)

    return tuple(worlds)


if __name__ == "__main__":
    data = read_grid(Path("examples/inputs/example_01d"))
