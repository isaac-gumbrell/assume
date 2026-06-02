# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Extract SRMC-run forecasts (prices + congestion signals) to a forecasts_df CSV.

Two-pass workflow
-----------------
1. Run an SRMC simulation (e.g. ``example_01j / base``) with ``log_flows: true`` in the
   market config.  This writes cleared prices to ``market_meta`` and per-line congestion
   to ``grid_flows`` in the database.

2. Run this script to pull those results and write (or merge) a ``forecasts_df.csv``
   ready for the RL learning pass:

   .. code-block:: console

       python scripts/extract_srmc_forecasts.py \\
           --db-uri sqlite:///./examples/local_db/assume_db.db \\
           --simulation-id example_01j_base \\
           --market-id EOM \\
           --start 2019-01-01 \\
           --end 2019-01-04 \\
           --freq 1h \\
           --output examples/inputs/example_01j/forecasts_df.csv

Output columns
--------------
* ``price_{market_id}``          — time-averaged cleared price per product hour
* ``{line_id}_congestion_signal`` — ``congestion_pct`` (flow / capacity) per line
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


def _load_prices(
    engine, simulation_id: str, market_id: str, index: pd.DatetimeIndex
) -> pd.Series | None:
    """Load per-hour cleared prices from *market_meta*."""
    query = text(
        "SELECT product_start, price FROM market_meta "
        "WHERE simulation = :sim AND market_id = :mid"
    )
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"sim": simulation_id, "mid": market_id})

    if df.empty:
        logger.warning(
            f"No market_meta rows found for simulation='{simulation_id}', market_id='{market_id}'"
        )
        return None

    df["product_start"] = pd.to_datetime(df["product_start"])
    # Average over nodes if multiple rows exist per timestep (nodal/zonal clearing)
    price_series = (
        df.groupby("product_start")["price"]
        .mean()
        .reindex(index, method="ffill")
        .bfill()
    )
    price_series.name = f"price_{market_id}"
    return price_series


def _load_congestion(
    engine, simulation_id: str, index: pd.DatetimeIndex
) -> pd.DataFrame | None:
    """Load per-line congestion_pct from *grid_flows*, pivot to wide format."""
    query = text(
        "SELECT datetime, line, congestion_pct FROM grid_flows WHERE simulation = :sim"
    )
    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"sim": simulation_id})
    except Exception as exc:
        logger.warning(f"Could not read grid_flows: {exc}")
        return None

    if df.empty:
        logger.warning(f"No grid_flows rows found for simulation='{simulation_id}'")
        return None

    if "congestion_pct" not in df.columns:
        logger.warning(
            "grid_flows table has no 'congestion_pct' column — "
            "re-run the SRMC simulation with the updated clearing code."
        )
        return None

    df["datetime"] = pd.to_datetime(df["datetime"])
    wide = (
        df.pivot_table(
            index="datetime", columns="line", values="congestion_pct", aggfunc="mean"
        )
        .reindex(index, method="ffill")
        .bfill()
    )
    wide.columns = [f"{col}_congestion_signal" for col in wide.columns]
    return wide


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def extract_srmc_forecasts(
    db_uri: str,
    simulation_id: str,
    market_id: str,
    start: str,
    end: str,
    freq: str,
    output_path: str,
    merge: bool = True,
) -> pd.DataFrame:
    """Extract prices + congestion signals and write ``forecasts_df.csv``.

    Args:
        db_uri: SQLAlchemy database URI (sqlite or postgresql).
        simulation_id: Simulation ID of the completed SRMC run.
        market_id: Market ID whose cleared prices to extract (e.g. ``"EOM"``).
        start: Scenario start date string (e.g. ``"2019-01-01"``).
        end: Scenario end date string (e.g. ``"2019-01-04"``).
        freq: Time step frequency string (e.g. ``"1h"``).
        output_path: Path to write (or update) ``forecasts_df.csv``.
        merge: If ``True`` and *output_path* already exists, merge new columns
            into the existing CSV rather than overwriting it.

    Returns:
        The combined DataFrame written to *output_path*.
    """
    engine = create_engine(db_uri)
    index = pd.date_range(start=start, end=end, freq=freq)

    frames: list[pd.DataFrame] = []

    # Price columns
    price = _load_prices(engine, simulation_id, market_id, index)
    if price is not None:
        frames.append(price.to_frame())

    # Congestion signal columns
    congestion = _load_congestion(engine, simulation_id, index)
    if congestion is not None:
        frames.append(congestion)

    if not frames:
        logger.error("Nothing extracted — output file not written.")
        return pd.DataFrame(index=index)

    result = pd.concat(frames, axis=1)
    result.index.name = "datetime"

    if merge:
        try:
            existing = pd.read_csv(output_path, index_col=0, parse_dates=True)
            # Overwrite columns that exist in both; keep others
            for col in result.columns:
                existing[col] = result[col]
            result = existing
        except FileNotFoundError:
            pass  # no existing file — write fresh

    result.to_csv(output_path, float_format="%.6g")
    logger.info(
        f"Wrote {len(result.columns)} columns × {len(result)} rows to '{output_path}'"
    )
    return result


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Extract SRMC prices + congestion signals to forecasts_df.csv"
    )
    p.add_argument("--db-uri", required=True, help="SQLAlchemy DB URI")
    p.add_argument(
        "--simulation-id", required=True, help="Simulation ID of the SRMC run"
    )
    p.add_argument(
        "--market-id",
        default="EOM",
        help="Market ID for price extraction (default: EOM)",
    )
    p.add_argument(
        "--start", required=True, help="Scenario start date (e.g. 2019-01-01)"
    )
    p.add_argument("--end", required=True, help="Scenario end date (e.g. 2019-01-04)")
    p.add_argument("--freq", default="1h", help="Time step frequency (default: 1h)")
    p.add_argument("--output", required=True, help="Output CSV path (forecasts_df.csv)")
    p.add_argument(
        "--no-merge",
        action="store_true",
        help="Overwrite output file instead of merging into existing",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    result = extract_srmc_forecasts(
        db_uri=args.db_uri,
        simulation_id=args.simulation_id,
        market_id=args.market_id,
        start=args.start,
        end=args.end,
        freq=args.freq,
        output_path=args.output,
        merge=not args.no_merge,
    )
    if result.empty:
        sys.exit(1)
