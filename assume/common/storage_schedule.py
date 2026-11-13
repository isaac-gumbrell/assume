# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exogenous storage charge/discharge schedules.

A schedule is a per-timestep power series for one storage unit, in MW, using the
unit's ``energy`` output sign convention: positive values discharge, negative values
charge. Schedules are supplied as ``storage_schedule_{unit_id}`` columns in
``forecasts_df.csv`` and are produced from a finished run with
:func:`export_storage_schedule`.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from assume.common.fast_pandas import FastIndex, FastSeries

if TYPE_CHECKING:
    from assume.common.base import BaseUnit

log = logging.getLogger(__name__)

SCHEDULE_COLUMN_PREFIX = "storage_schedule_"


def schedule_column(unit_id: str) -> str:
    """Return the ``forecasts_df`` column name holding the schedule of ``unit_id``."""
    return f"{SCHEDULE_COLUMN_PREFIX}{unit_id}"


def replay_schedule_soc(
    schedule: np.ndarray,
    initial_soc: float,
    capacity: float,
    efficiency_charge: float,
    efficiency_discharge: float,
    duration_hours: float,
    min_soc: float = 0.0,
    max_soc: float = 1.0,
    soc_tolerance: float = 0.0,
) -> np.ndarray:
    """Replay the state of charge implied by ``schedule``.

    Uses the same update as :meth:`assume.units.storage.Storage.execute_current_dispatch`.
    An overshoot of ``[min_soc, max_soc]`` by at most ``soc_tolerance`` is reset to
    the bound, as the unit does when it executes the dispatch, so rounding in an
    exported schedule does not accumulate. Larger overshoots are kept in the result.

    Returns:
        np.ndarray: SoC fractions with length ``len(schedule) + 1``. Entry ``t`` is the
        SoC at the start of step ``t``.
    """
    schedule = np.asarray(schedule, dtype=float)
    soc_change = (
        np.where(
            schedule > 0,
            -schedule * duration_hours / efficiency_discharge,
            -schedule * duration_hours * efficiency_charge,
        )
        / capacity
    )
    soc = np.empty(len(schedule) + 1)
    soc[0] = initial_soc
    for t, change in enumerate(soc_change):
        next_soc = soc[t] + change
        if min_soc - soc_tolerance <= next_soc < min_soc:
            next_soc = min_soc
        elif max_soc < next_soc <= max_soc + soc_tolerance:
            next_soc = max_soc
        soc[t + 1] = next_soc
    return soc


def validate_storage_schedule(
    unit: BaseUnit, schedule: np.ndarray, tolerance: float
) -> None:
    """Raise a ValueError if ``schedule`` is physically infeasible for ``unit``.

    Checks for missing values, power limits (scaled by availability), the
    minimum-power deadband, and the SoC bounds when replayed from ``initial_soc``.
    Ramp limits depend on dispatch history and are checked at bid time.

    Args:
        unit: The storage unit.
        schedule: Power per step of ``unit.index`` in MW (+ discharge, - charge).
        tolerance: Absolute tolerance in MW. The SoC tolerance is
            ``tolerance * duration / capacity``.
    """
    schedule = np.asarray(schedule, dtype=float)
    times = unit.index.get_date_list()
    if len(schedule) != len(times):
        raise ValueError(
            f"Storage schedule for '{unit.id}' has {len(schedule)} steps, expected "
            f"{len(times)}."
        )

    def fail(position: int, message: str):
        raise ValueError(
            f"Infeasible storage schedule for '{unit.id}' at {times[position]}: "
            f"{message}"
        )

    missing = np.flatnonzero(np.isnan(schedule))
    if missing.size:
        fail(missing[0], "value is missing.")

    availability = np.asarray(unit.forecaster.availability.data, dtype=float)
    max_discharge = availability * unit.max_power_discharge
    max_charge = availability * unit.max_power_charge
    over_discharge = np.flatnonzero(schedule > max_discharge + tolerance)
    if over_discharge.size:
        t = over_discharge[0]
        fail(t, f"discharge {schedule[t]} MW exceeds {max_discharge[t]} MW.")
    over_charge = np.flatnonzero(schedule < max_charge - tolerance)
    if over_charge.size:
        t = over_charge[0]
        fail(t, f"charge {schedule[t]} MW exceeds {max_charge[t]} MW.")

    in_discharge_deadband = np.flatnonzero(
        (schedule > tolerance) & (schedule < unit.min_power_discharge - tolerance)
    )
    if in_discharge_deadband.size:
        t = in_discharge_deadband[0]
        fail(
            t,
            f"discharge {schedule[t]} MW is below min_power_discharge "
            f"{unit.min_power_discharge} MW.",
        )
    in_charge_deadband = np.flatnonzero(
        (schedule < -tolerance) & (schedule > unit.min_power_charge + tolerance)
    )
    if in_charge_deadband.size:
        t = in_charge_deadband[0]
        fail(
            t,
            f"charge {schedule[t]} MW is below min_power_charge "
            f"{unit.min_power_charge} MW.",
        )

    duration_hours = unit.index.freq / timedelta(hours=1)
    soc_tolerance = tolerance * duration_hours / unit.capacity
    soc = replay_schedule_soc(
        schedule,
        unit.initial_soc,
        unit.capacity,
        unit.efficiency_charge,
        unit.efficiency_discharge,
        duration_hours,
        unit.min_soc,
        unit.max_soc,
        soc_tolerance,
    )
    out_of_bounds = np.flatnonzero((soc[1:] < unit.min_soc) | (soc[1:] > unit.max_soc))
    if out_of_bounds.size:
        t = out_of_bounds[0]
        fail(
            t,
            f"SoC after this step is {soc[t + 1]:.6f}, outside "
            f"[{unit.min_soc}, {unit.max_soc}] (replayed from initial_soc "
            f"{unit.initial_soc}).",
        )


def load_storage_schedule(
    forecast_df: pd.DataFrame | None,
    units,
    initializing_unit: BaseUnit | None,
    index: FastIndex,
) -> FastSeries | None:
    """Return the schedule of ``initializing_unit`` from ``forecast_df``.

    Raises a ValueError if any schedule column names a unit that is not a storage
    unit of this simulation. Returns None when the unit has no schedule column, so
    strategies that require a schedule can raise their own error.
    """
    if not isinstance(forecast_df, pd.DataFrame) or initializing_unit is None:
        return None

    columns = [
        col for col in forecast_df.columns if col.startswith(SCHEDULE_COLUMN_PREFIX)
    ]
    if not columns:
        return None

    from assume.units.storage import Storage

    storage_ids = {unit.id for unit in units if isinstance(unit, Storage)}
    unknown = sorted(
        col for col in columns if col[len(SCHEDULE_COLUMN_PREFIX) :] not in storage_ids
    )
    if unknown:
        raise ValueError(
            f"forecasts_df has storage schedule columns for unknown storage units: "
            f"{unknown}."
        )

    column = schedule_column(initializing_unit.id)
    if column not in forecast_df.columns:
        return None

    series = forecast_df[column].reindex(index.as_datetimeindex())
    return FastSeries(index=index, value=series.to_numpy(dtype=float), name=column)


def schedule_from_unit_dispatch(
    unit_dispatch: pd.DataFrame,
    index: pd.DatetimeIndex,
    storage_unit_ids: list[str] | None = None,
    tolerance: float = 1e-4,
) -> pd.DataFrame:
    """Build schedule columns from a run's ``unit_dispatch`` output.

    Timestamps of ``index`` after the last dispatched step get zero (no market
    product covered them). Gaps inside the dispatched period raise an error. The
    recorded SoC of every unit is checked against the SoC replayed from the
    exported power, so a schedule that does not reproduce the run is rejected.

    Args:
        unit_dispatch: Long table with columns ``time`` (or a time index), ``unit``,
            ``power`` and ``soc``.
        index: Simulation index of the runs that will read the schedule.
        storage_unit_ids: Units to export. Defaults to every unit with a SoC.
        tolerance: Absolute SoC tolerance for the replay check.

    Returns:
        pd.DataFrame: ``storage_schedule_{unit_id}`` columns indexed by ``index``.
    """
    dispatch = unit_dispatch.reset_index()
    if "time" not in dispatch.columns:
        raise ValueError("unit_dispatch must contain a 'time' column or index.")
    missing_columns = {"unit", "power", "soc"} - set(dispatch.columns)
    if missing_columns:
        raise ValueError(f"unit_dispatch is missing columns {sorted(missing_columns)}.")
    dispatch["time"] = pd.to_datetime(dispatch["time"])

    if storage_unit_ids is None:
        storage_unit_ids = sorted(
            dispatch.loc[dispatch["soc"].notna(), "unit"].unique()
        )
    if not storage_unit_ids:
        raise ValueError("unit_dispatch contains no storage units.")
    unknown = sorted(set(storage_unit_ids) - set(dispatch["unit"]))
    if unknown:
        raise ValueError(f"Storage units {unknown} are not in unit_dispatch.")

    dispatch = dispatch[dispatch["unit"].isin(storage_unit_ids)]
    duplicated = dispatch.duplicated(subset=["time", "unit"])
    if duplicated.any():
        first = dispatch[duplicated].iloc[0]
        raise ValueError(
            f"unit_dispatch has duplicate rows, e.g. unit '{first['unit']}' at "
            f"{first['time']}."
        )

    power = dispatch.pivot(index="time", columns="unit", values="power")
    soc = dispatch.pivot(index="time", columns="unit", values="soc")
    outside = power.index.difference(index)
    if len(outside):
        raise ValueError(
            f"unit_dispatch has {len(outside)} timestamps outside the target index, "
            f"first {outside[0]}."
        )

    last_dispatched = power.index.max()
    within = index[index <= last_dispatched]
    power = power.reindex(within)
    soc = soc.reindex(within)
    if power.isna().any().any():
        unit_id = power.columns[power.isna().any()][0]
        gap = power.index[power[unit_id].isna()][0]
        raise ValueError(f"unit_dispatch has no power for '{unit_id}' at {gap}.")

    for unit_id in power.columns:
        recorded = soc[unit_id].to_numpy(dtype=float)
        if np.isnan(recorded).any():
            gap = soc.index[np.isnan(recorded)][0]
            raise ValueError(f"unit_dispatch has no SoC for '{unit_id}' at {gap}.")
        steps = power[unit_id].to_numpy(dtype=float)
        # Unit parameters are not part of unit_dispatch, so only the direction of
        # each SoC change is checked here. The full SoC replay runs when the
        # schedule is loaded into a simulation.
        changes = np.diff(recorded)
        discharging = steps[:-1] > 0
        charging = steps[:-1] < 0
        idle_change = np.abs(changes[~discharging & ~charging])
        if idle_change.size and idle_change.max() > tolerance:
            raise ValueError(
                f"Recorded SoC of '{unit_id}' changes in a step with zero power; "
                "unit_dispatch is not a consistent storage trajectory."
            )
        if (changes[discharging] > tolerance).any() or (
            changes[charging] < -tolerance
        ).any():
            raise ValueError(
                f"Recorded SoC of '{unit_id}' moves against the sign of its power; "
                "unit_dispatch is not a consistent storage trajectory."
            )

    schedule = power.reindex(index, fill_value=0.0)
    schedule.columns = [schedule_column(unit_id) for unit_id in schedule.columns]
    schedule.index.name = "datetime"
    trailing = len(index) - len(within)
    if trailing:
        log.info(
            "Storage schedule: %d trailing timestamps after %s have no dispatch and "
            "are set to 0.",
            trailing,
            last_dispatched,
        )
    return schedule


def read_unit_dispatch(
    simulation_id: str, db_uri: str | None = None, csv_path: str | Path | None = None
) -> pd.DataFrame:
    """Read the ``unit_dispatch`` output of ``simulation_id`` from a DB or CSV export.

    Exactly one of ``db_uri`` and ``csv_path`` must be given. ``csv_path`` is the
    ``export_csv_path`` used by the run (the file read is
    ``{csv_path}/{simulation_id}/unit_dispatch.csv``).
    """
    if (db_uri is None) == (csv_path is None):
        raise ValueError("Give exactly one of db_uri and csv_path.")

    if csv_path is not None:
        file_path = Path(csv_path) / simulation_id / "unit_dispatch.csv"
        if not file_path.is_file():
            raise FileNotFoundError(f"{file_path} does not exist.")
        dispatch = pd.read_csv(file_path, parse_dates=["time"])
        log.warning(
            "CSV outputs store 5 significant digits. Schedules exported from %s "
            "carry this rounding; set schedule_tolerance (MW) above it, or export "
            "from the database for full precision.",
            file_path,
        )
    else:
        from sqlalchemy import create_engine, text

        engine = create_engine(db_uri)
        try:
            with engine.connect() as conn:
                dispatch = pd.read_sql(
                    text(
                        "SELECT time, unit, power, soc FROM unit_dispatch"
                        " WHERE simulation = :sim"
                    ),
                    conn,
                    params={"sim": simulation_id},
                    parse_dates=["time"],
                )
        finally:
            engine.dispose()

    if "simulation" in dispatch.columns:
        dispatch = dispatch[dispatch["simulation"] == simulation_id]
    if dispatch.empty:
        raise ValueError(f"No unit_dispatch rows for simulation '{simulation_id}'.")
    return dispatch


def export_storage_schedule(
    simulation_id: str,
    index: pd.DatetimeIndex,
    db_uri: str | None = None,
    csv_path: str | Path | None = None,
    storage_unit_ids: list[str] | None = None,
    output_file: str | Path | None = None,
) -> pd.DataFrame:
    """Export the realised storage dispatch of a run as schedule columns.

    The returned columns can be added to ``forecasts_df.csv`` of later runs. If
    ``output_file`` is given, the schedule is also written there as CSV.
    """
    dispatch = read_unit_dispatch(simulation_id, db_uri=db_uri, csv_path=csv_path)
    schedule = schedule_from_unit_dispatch(dispatch, index, storage_unit_ids)
    if output_file is not None:
        schedule.to_csv(output_file)
    return schedule
