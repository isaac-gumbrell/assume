# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from datetime import timedelta

import numpy as np
import pandas as pd
import pypsa
from linopy import available_solvers

from assume.common.market_objects import MarketProduct
from assume.common.utils import SUPPORTED_SOLVERS

logger = logging.getLogger(__name__)


def add_generators(
    network: pypsa.Network,
    generators: pd.DataFrame,
) -> None:
    """
    Add generators normally to the grid

    Args:
        network (pypsa.Network): the pypsa network to which the generators are
        generators (pandas.DataFrame): the generators dataframe
    """
    p_set = pd.DataFrame(
        np.zeros((len(network.snapshots), len(generators.index))),
        index=network.snapshots,
        columns=generators.index,
    )

    if isinstance(generators, dict):
        gen_c = generators.copy()

        if "p_min_pu" not in gen_c.columns:
            gen_c["p_min_pu"] = p_set
        if "p_max_pu" not in gen_c.columns:
            gen_c["p_max_pu"] = p_set + 1
        if "marginal_cost" not in gen_c.columns:
            gen_c["marginal_cost"] = p_set

        network.add(
            "Generator",
            name=generators.index,
            bus=generators["node"],  # bus to which the generator is connected to
            p_nom=generators[
                "max_power"
            ],  # Nominal capacity of the powerplant/generator
            **gen_c,
        )
    else:
        # add generators
        generators.drop(
            ["p_min_pu", "p_max_pu", "marginal_cost"],
            axis=1,
            inplace=True,
            errors="ignore",
        )
        network.add(
            "Generator",
            name=generators.index,
            bus=generators["node"],  # bus to which the generator is connected to
            p_nom=generators[
                "max_power"
            ],  # Nominal capacity of the powerplant/generator
            p_min_pu=p_set,
            p_max_pu=p_set + 1,
            marginal_cost=p_set,
            **generators,
        )


def add_redispatch_generators(
    network: pypsa.Network,
    generators: pd.DataFrame,
    backup_marginal_cost: float = 1e5,
) -> None:
    """
    Adds the given generators for redispatch.
    This includes functions to optimize up as well as down and adds backup capacities of powerplants to be able to adjust accordingly when a congestion happens.

    Args:
        network (pypsa.Network): the pypsa network to which the generators are
        generators (pandas.DataFrame): the generators dataframe
        backup_marginal_cost (float, optional): The cost of dispatching the backup units in [€/MW]. Defaults to 1e5.
    """
    p_set = pd.DataFrame(
        np.zeros((len(network.snapshots), len(generators.index))),
        index=network.snapshots,
        columns=generators.index,
    )

    # add generators and their sold capacities as load with reversed sign to have fixed feed in
    network.add(
        "Load",
        name=generators.index,
        bus=generators["node"],  # bus to which the generator is connected to
        p_set=p_set,
        sign=1,
    )

    # add upward redispatch generators
    network.add(
        "Generator",
        name=generators.index,
        suffix="_up",
        bus=generators["node"],  # bus to which the generator is connected to
        p_nom=generators["max_power"],  # Nominal capacity of the powerplant/generator
        p_min_pu=p_set,
        p_max_pu=p_set + 1,
        marginal_cost=p_set,
    )

    # add downward redispatch generators
    network.add(
        "Generator",
        name=generators.index,
        suffix="_down",
        bus=generators["node"],  # bus to which the generator is connected to
        p_nom=generators["max_power"],  # Nominal capacity of the powerplant/generator
        p_min_pu=p_set,
        p_max_pu=p_set + 1,
        marginal_cost=p_set,
        sign=-1,
    )

    # add upward and downward backup generators at each node
    network.add(
        "Generator",
        name=network.buses.index,
        suffix="_backup_up",
        bus=network.buses.index,  # bus to which the generator is connected to
        p_nom=10e4,
        marginal_cost=backup_marginal_cost,
    )

    network.add(
        "Generator",
        name=network.buses.index,
        suffix="_backup_down",
        bus=network.buses.index,  # bus to which the generator is connected to
        p_nom=10e4,
        marginal_cost=backup_marginal_cost,
        sign=-1,
    )


def add_backup_generators(
    network: pypsa.Network,
    backup_marginal_cost: float = 1e5,
) -> None:
    """
    Add generators normally to the grid

    Args:
        network (pypsa.Network): the pypsa network to which the generators are
        generators (pandas.DataFrame): the generators dataframe
    """

    # add backup generators at each node
    network.add(
        "Generator",
        name=network.buses.index,
        suffix="_backup",
        bus=network.buses.index,  # bus to which the generator is connected to
        p_nom=10e4,
        marginal_cost=backup_marginal_cost,
    )


def add_loads(
    network: pypsa.Network,
    loads: pd.DataFrame,
) -> None:
    """
    Add loads normally to the grid

    Args:
        network (pypsa.Network): the pypsa network to which the loads are
        loads (pandas.DataFrame): the loads dataframe
    """

    # add loads
    network.add(
        "Load",
        name=loads.index,
        bus=loads["node"],  # bus to which the generator is connected to
        **loads,
    )

    if "p_set" not in loads.columns:
        network.loads_t["p_set"] = pd.DataFrame(
            np.zeros((len(network.snapshots), len(loads.index))),
            index=network.snapshots,
            columns=loads.index,
        )


def add_redispatch_loads(
    network: pypsa.Network,
    loads: pd.DataFrame,
) -> None:
    """
    This adds loads to the redispatch PyPSA network with respective bus data to which they are connected
    """
    loads_c = loads.copy()
    if "sign" in loads_c.columns:
        del loads_c["sign"]

    # add loads with opposite sign (default for loads is -1). This is needed to properly model the redispatch
    network.add(
        "Load",
        name=loads.index,
        bus=loads["node"],  # bus to which the generator is connected to
        sign=1,
        **loads_c,
    )

    if "p_set" not in loads.columns:
        network.loads_t["p_set"] = pd.DataFrame(
            np.zeros((len(network.snapshots), len(loads.index))),
            index=network.snapshots,
            columns=loads.index,
        )


def add_nodal_loads(
    network: pypsa.Network,
    loads: pd.DataFrame,
) -> None:
    """
    This adds loads to the nodal PyPSA network with respective bus data to which they are connected.
    The loads are added as generators with negative sign so their dispatch can be also curtailed,
    since regular load in PyPSA represents only an inelastic demand.
    """
    p_set = pd.DataFrame(
        np.zeros((len(network.snapshots), len(loads.index))),
        index=network.snapshots,
        columns=loads.index,
    )
    loads_c = loads.copy()

    if "sign" in loads_c.columns:
        del loads_c["sign"]

    # add loads as negative generators
    network.add(
        "Generator",
        name=loads.index,
        bus=loads["node"],  # bus to which the generator is connected to
        p_nom=loads["max_power"],  # Nominal capacity of the powerplant/generator
        p_min_pu=p_set,
        p_max_pu=p_set + 1,
        marginal_cost=p_set,
        sign=-1,
        **loads_c,
    )


def read_pypsa_grid(
    network: pypsa.Network,
    grid_dict: dict[str, pd.DataFrame],
):
    """
    Generates the pypsa grid from a grid dictionary.
    Does not add the generators, as they are added in different ways, depending on whether redispatch is used.

    Args:
        network (pypsa.Network): the pypsa network to which the components will be added
        grid_dict (dict[str, pd.DataFrame]): the dictionary containing dataframes for generators, loads, buses and links
    """

    def add_buses(network: pypsa.Network, buses: pd.DataFrame) -> None:
        network.add("Bus", buses.index, **buses)

    def add_lines(network: pypsa.Network, lines: pd.DataFrame) -> None:
        network.add("Line", lines.index, **lines)

    # setup the network
    add_buses(network, grid_dict["buses"])
    add_lines(network, grid_dict["lines"])
    network.add("Carrier", "AC")
    return network


def calculate_network_meta(network, product: MarketProduct, i: int):
    """
    This function calculates the meta data such as supply and demand volumes, and nodal prices.

    Args:
        product (MarketProduct): The product for which clearing happens.
        i (int): The index of the product in the market products list.

    Returns:
        dict: The meta data.
    """

    meta = []
    duration_hours = (product[1] - product[0]) / timedelta(hours=1)
    # iterate over buses
    for bus in network.buses.index:
        # add backup dispatch to dispatch
        # Step 1: Identify generators connected to the specified bus
        generators_connected_to_bus = network.generators[
            network.generators.bus == bus
        ].index

        # Step 2: Select dispatch levels for these generators from network.generators_t.p
        dispatch_for_bus = network.generators_t.p[generators_connected_to_bus].iloc[i]
        # multiple by network.generators.sign to get the correct sign for dispatch
        dispatch_for_bus = (
            dispatch_for_bus * network.generators.sign[generators_connected_to_bus]
        )

        supply_volume = dispatch_for_bus[dispatch_for_bus > 0].sum()
        demand_volume = -dispatch_for_bus[dispatch_for_bus < 0].sum()
        if not network.buses_t.marginal_price.empty:
            price = network.buses_t.marginal_price[str(bus)].iat[i]
        else:
            price = 0

        meta.append(
            {
                "supply_volume": supply_volume,
                "demand_volume": demand_volume,
                "demand_volume_energy": demand_volume * duration_hours,
                "supply_volume_energy": supply_volume * duration_hours,
                "price": price,
                "node": bus,
                "product_start": product[0],
                "product_end": product[1],
                "only_hours": product[2],
            }
        )

    return meta


def compute_flows_congestion_pct(
    flow_df: pd.DataFrame, lines: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute per-line congestion percentage from a wide-format flows DataFrame.

    ``congestion_pct = flow / capacity``, where capacity is direction-aware:

    - ``flow > 0``  → ``s_nom_forward`` (bus0 → bus1), if present and valid
    - ``flow ≤ 0``  → ``s_nom_reverse`` (bus1 → bus0), if present and valid
    - Falls back to ``s_nom * s_max_pu`` (or ``s_nom``) for lines without
      directional columns or with NaN / zero directional values.

    The OPF guarantees values are bounded to ``[-1, 1]``.  Lines that are not
    found in *lines* are assigned ``0.0``.

    Args:
        flow_df: DataFrame with index = datetime, columns = line_id, values = MW flow.
        lines:   DataFrame with index = line_id, at minimum a ``s_nom`` column.
                 Optionally ``s_nom_forward``, ``s_nom_reverse``, ``s_max_pu``.

    Returns:
        DataFrame with the same shape as *flow_df*, values in ``[-1, 1]``.
    """
    has_directional = (
        "s_nom_forward" in lines.columns and "s_nom_reverse" in lines.columns
    )

    congestion_df = pd.DataFrame(0.0, index=flow_df.index, columns=flow_df.columns)

    for line_id in flow_df.columns:
        if line_id not in lines.index:
            continue

        flow_vals = flow_df[line_id].values

        if has_directional:
            cap_f = lines.at[line_id, "s_nom_forward"]
            cap_r = lines.at[line_id, "s_nom_reverse"]
            if not pd.isna(cap_f) and not pd.isna(cap_r) and cap_f > 0 and cap_r > 0:
                congestion_df[line_id] = np.where(
                    flow_vals > 0, flow_vals / cap_f, flow_vals / cap_r
                )
                continue

        # Symmetric fallback
        s_max_pu = (
            lines.at[line_id, "s_max_pu"]
            if "s_max_pu" in lines.columns
            and not pd.isna(lines.at[line_id, "s_max_pu"])
            else 1.0
        )
        capacity = lines.at[line_id, "s_nom"] * s_max_pu
        if capacity != 0:
            congestion_df[line_id] = flow_vals / capacity

    return congestion_df


def get_supported_solver_linopy(default_solver: str | None = None):
    """
    Get an available solver for linopy optimization.

    Filters the list of supported solvers to find which ones are installed,
    then returns the default solver if available, otherwise falls back to the first available solver.

    Args:
        default_solver (str | None, optional): Preferred solver name. If not available,
            falls back to the first available solver. Defaults to None.

    Returns:
        str: Name of the selected solver.

    Raises:
        RuntimeError: If none of the supported solvers (highs, gurobi, glpk, cbc, cplex) are available.

    Warning:
        Logs a warning if the default_solver is not available and a fallback is used.
    """
    solvers_priority = SUPPORTED_SOLVERS

    # Filter available solvers while preserving the shared fallback priority.
    solvers = [solver for solver in solvers_priority if solver in available_solvers]
    if not solvers:
        raise RuntimeError(f"None of {solvers_priority} are available")

    solver = default_solver or solvers[0]

    if solver not in solvers:
        logger.warning("Solver %s not available, using %s", solver, solvers[0])
        solver = solvers[0]

    return solver
