# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime, timedelta

import numpy as np

from assume.common.base import MinMaxChargeStrategy, SupportsMinMaxCharge
from assume.common.fast_pandas import FastSeries
from assume.common.market_objects import MarketConfig, Orderbook, Product
from assume.common.storage_schedule import (
    replay_schedule_soc,
    schedule_column,
    validate_storage_schedule,
)
from assume.strategies.flexable_storage import StorageEnergyHeuristicFlexableStrategy


class StorageScheduleMixin:
    """Shared handling of exogenous storage schedules.

    The schedule of a unit is read from ``unit.forecaster.storage_schedule``
    (``storage_schedule_{unit_id}`` columns of forecasts_df) and validated once per
    loaded series. Foreign units in staggered training may omit their column; they
    are treated as idle.
    """

    def _init_schedule_params(
        self,
        schedule_discharge_price: float | None,
        schedule_charge_price: float | None,
        schedule_tolerance: float,
    ) -> None:
        for name, value in (
            ("schedule_discharge_price", schedule_discharge_price),
            ("schedule_charge_price", schedule_charge_price),
        ):
            if value is None:
                raise ValueError(
                    f"{type(self).__name__} requires '{name}' in "
                    "bidding_strategy_params."
                )
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}.")
        if not np.isfinite(schedule_tolerance) or schedule_tolerance < 0:
            raise ValueError(
                "schedule_tolerance must be a finite non-negative number, got "
                f"{schedule_tolerance}."
            )
        self.schedule_discharge_price = float(schedule_discharge_price)
        self.schedule_charge_price = float(schedule_charge_price)
        self.schedule_tolerance = float(schedule_tolerance)
        self._validated_schedules: dict[str, FastSeries] = {}

    def _unit_schedule(self, unit: SupportsMinMaxCharge) -> FastSeries:
        schedule = unit.forecaster.storage_schedule
        if schedule is None:
            if not unit.forecaster.is_foreign:
                raise ValueError(
                    f"{type(self).__name__} requires column "
                    f"'{schedule_column(unit.id)}' in forecasts_df.csv for unit "
                    f"'{unit.id}' (and use_forecasts_df must not be false)."
                )
            schedule = FastSeries(index=unit.forecaster.index, value=0.0)
            unit.forecaster.storage_schedule = schedule

        if self._validated_schedules.get(unit.id) is not schedule:
            validate_storage_schedule(unit, schedule.data, self.schedule_tolerance)
            self._validated_schedules[unit.id] = schedule
        return schedule

    def _schedule_soc(
        self, unit: SupportsMinMaxCharge, schedule: FastSeries
    ) -> FastSeries:
        """Return the SoC at the start of each step if ``schedule`` is executed."""
        duration_hours = unit.index.freq / timedelta(hours=1)
        soc = replay_schedule_soc(
            schedule.data,
            unit.initial_soc,
            unit.capacity,
            unit.efficiency_charge,
            unit.efficiency_discharge,
            duration_hours,
            unit.min_soc,
            unit.max_soc,
            self.schedule_tolerance * duration_hours / unit.capacity,
        )
        return FastSeries(index=unit.index, value=soc[:-1])

    def _scheduled_power(self, unit: SupportsMinMaxCharge, start: datetime) -> float:
        power = float(self._unit_schedule(unit).at[start])
        return 0.0 if abs(power) <= self.schedule_tolerance else power

    @staticmethod
    def _feasible_power_limits(
        unit: SupportsMinMaxCharge,
        start: datetime,
        end: datetime,
        soc: float,
        previous_power: float,
    ) -> tuple[float, float]:
        """Return (max discharge >= 0, max charge <= 0) at ``soc`` incl. ramping."""
        current_power = unit.outputs["energy"].at[start]
        if current_power != 0:
            raise ValueError(
                f"Storage schedules support a single energy market, but unit "
                f"'{unit.id}' already has {current_power} MW at {start}."
            )
        min_discharge, max_discharge = unit.calculate_min_max_discharge(
            start, end, soc=soc
        )
        min_charge, max_charge = unit.calculate_min_max_charge(start, end, soc=soc)
        max_discharge = unit.calculate_ramp_discharge(
            soc, previous_power, max_discharge[0], 0, min_discharge[0]
        )
        max_charge = unit.calculate_ramp_charge(
            soc, previous_power, max_charge[0], 0, min_charge[0]
        )
        return float(max_discharge), float(max_charge)

    @staticmethod
    def _soc_after(
        unit: SupportsMinMaxCharge, soc: float, power: float, duration_hours: float
    ) -> float:
        if power > 0:
            return (
                soc - power * duration_hours / unit.efficiency_discharge / unit.capacity
            )
        return soc - power * duration_hours * unit.efficiency_charge / unit.capacity

    def _schedule_bid(
        self,
        unit: SupportsMinMaxCharge,
        start: datetime,
        end: datetime,
        volume: float,
    ) -> dict:
        return {
            "start_time": start,
            "end_time": end,
            "only_hours": None,
            "price": (
                self.schedule_discharge_price
                if volume > 0
                else self.schedule_charge_price
            ),
            "volume": volume,
            "node": unit.node,
        }


class StorageEnergyFixedScheduleStrategy(StorageScheduleMixin, MinMaxChargeStrategy):
    """Bid an exogenous storage schedule as price-taking orders.

    Scheduled discharge is offered at ``schedule_discharge_price`` and scheduled
    charging is bid at ``schedule_charge_price``. Choose the discharge price at or
    below the cheapest generator offer and the charge price below the inflexible
    demand bid, so that scheduled volumes clear without displacing demand.

    The schedule is strict: infeasibility at bid time or a cleared volume that
    differs from the schedule raises a ValueError.

    Args:
        schedule_discharge_price (float): Offer price for scheduled discharge (required).
        schedule_charge_price (float): Bid price for scheduled charging (required).
        schedule_tolerance (float): Absolute tolerance in MW. Defaults to 0.01.
    """

    def __init__(
        self,
        *args,
        schedule_discharge_price: float | None = None,
        schedule_charge_price: float | None = None,
        schedule_tolerance: float = 0.01,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._init_schedule_params(
            schedule_discharge_price, schedule_charge_price, schedule_tolerance
        )

    def calculate_bids(
        self,
        unit: SupportsMinMaxCharge,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        first_start = product_tuples[0][0]
        soc = unit.outputs["soc"].at[first_start]
        previous_power = unit.get_output_before(first_start)

        bids = []
        for product in product_tuples:
            start, end = product[0], product[1]
            power = self._scheduled_power(unit, start)
            if power != 0:
                max_discharge, max_charge = self._feasible_power_limits(
                    unit, start, end, soc, previous_power
                )
                if (
                    power > max_discharge + self.schedule_tolerance
                    or power < max_charge - self.schedule_tolerance
                ):
                    raise ValueError(
                        f"Storage schedule for '{unit.id}' at {start} requests "
                        f"{power} MW, but only [{max_charge}, {max_discharge}] MW is "
                        f"feasible at SoC {soc:.6f} after {previous_power} MW."
                    )
                bids.append(self._schedule_bid(unit, start, end, power))

            duration_hours = (end - start) / timedelta(hours=1)
            soc = self._soc_after(unit, soc, power, duration_hours)
            previous_power = power

        return bids

    def calculate_reward(
        self,
        unit: SupportsMinMaxCharge,
        marketconfig: MarketConfig,
        orderbook: Orderbook,
    ):
        for order in orderbook:
            if "accepted_volume" not in order:
                raise ValueError(
                    f"Order of '{unit.id}' at {order['start_time']} has no "
                    "accepted_volume."
                )
            accepted = order["accepted_volume"]
            if abs(accepted - order["volume"]) > self.schedule_tolerance:
                raise ValueError(
                    f"Scheduled volume of '{unit.id}' did not clear at "
                    f"{order['start_time']} (node {order.get('node')}): scheduled "
                    f"{order['volume']} MW, accepted {accepted} MW. The schedule is "
                    "not feasible in this market or network."
                )
        StorageEnergyHeuristicFlexableStrategy.calculate_reward(
            self, unit, marketconfig, orderbook
        )
