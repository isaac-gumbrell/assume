# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Smoke driver for D3 paired-scenario MATD3 staggered training.

Runs the minimal `staggered_bau` / `staggered_inv` example pair through
``run_staggered_learning`` and writes results to a per-run SQLite database.

Usage (Windows PowerShell):
    .\\.venv\\Scripts\\python.exe examples\\world_script_staggered.py
"""

from __future__ import annotations

import logging
import os

from assume.scenario.loader_csv import run_staggered_learning


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    inputs_path = "examples/inputs"
    primary_scenario = "staggered_bau"
    study_case = "staggered"

    db_path = "examples/local_db/assume_staggered.db"
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    db_uri = f"sqlite:///{db_path}"

    csv_path = "examples/outputs/staggered"
    os.makedirs(csv_path, exist_ok=True)

    run_staggered_learning(
        inputs_path=inputs_path,
        scenario=primary_scenario,
        study_case=study_case,
        db_uri=db_uri,
        export_csv_path=csv_path,
        log_level="INFO",
        verbose=True,
    )


if __name__ == "__main__":
    main()
