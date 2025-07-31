import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression

import pypsa
from pypsa.descriptors import get_switchable_as_dense as get_as_dense
from pypsa.optimization.compat import define_constraints, get_var
from src.utils import mycmap, mycmap_dark 
from src.utils import create_dir

# -------------------------
# Utility Functions
# -------------------------
def annuity(capex, lifetime, wacc):
    """
    Calculate the annuity factor for a given capital expenditure, lifetime, and weighted average cost of capital (WACC).

    Args:
        capex (float): Capital expenditure (investment cost).
        lifetime (int): Lifetime of the asset in years.
        wacc (float): Weighted average cost of capital (as a decimal, e.g., 0.05 for 5%).

    Returns:
        df (pd.DataFrame): Annualized cost of the asset.
    """
    return capex * (wacc * (1 + wacc)**lifetime) / ((1 + wacc)**lifetime - 1)

def resolve_attribute(constant, timeseries_key, df_timeseries):
    """
    Resolve an attribute that can either be a constant or a timeseries.

    Args:
        constant: A constant value or NaN.
        timeseries_key: The key for the timeseries in the DataFrame.
        df_timeseries (pd.DataFrame): DataFrame containing timeseries data.

    Returns:
        df (pd.Series): The resolved timeseries or constant value.
    """
    return df_timeseries.loc[:, timeseries_key] if pd.isna(constant) else constant

# -------------------------
# Timeseries Attribute Assignment
# -------------------------
def add_effs_and_marginalcosts(index, row, df_timeseries):
    """
    Resolve efficiency and marginal costs for a component based on its attributes.

    Args:
        index (Any): Index of the component.
        row (pd.Series): Row from the DataFrame containing component attributes.
        df_timeseries (pd.DataFrame): DataFrame containing timeseries data.

    Returns:
        fuel (pd.Series): Fuel cost timeseries.
        eff (Union[pd.Series, float]): Efficiency timeseries or constant value.
        p_max_pu (Union[pd.Series, float]): Maximum power per unit (pu) timeseries or constant value.
        p_min_pu (Union[pd.Series, float]): Minimum power per unit (pu) timeseries or constant value.
    """

    if not pd.isna(row.fuel_costs) and not pd.isna(row.fuel_costs_ts):
        raise ValueError(f"Component {index} has both 'fuel_costs' and 'fuel_costs_ts' defined.")
    if not pd.isna(row.efficiency) and not pd.isna(row.efficiency_ts):
        raise ValueError(f"Component {index} has both 'efficiency' and 'efficiency_ts' defined.")
    if not pd.isna(row.p_max_pu) and not pd.isna(row.p_max_pu_ts):
        raise ValueError(f"Component {index} has both 'p_max_pu' and 'p_max_pu_ts' defined.")
    if not pd.isna(row.p_min_pu) and not pd.isna(row.p_min_pu_ts):
        raise ValueError(f"Component {index} has both 'p_min_pu' and 'p_min_pu_ts' defined.")

    if not pd.isna(row.efficiency) and not pd.isna(row.fuel_costs):
        eff, fuel = row.efficiency, row.fuel_costs
    elif not pd.isna(row.fuel_costs) and not pd.isna(row.efficiency_ts):
        fuel, eff = row.fuel_costs, df_timeseries.loc[:, row.efficiency_ts]
    elif not pd.isna(row.fuel_costs_ts) and not pd.isna(row.efficiency):
        fuel, eff = df_timeseries.loc[:, row.fuel_costs_ts], row.efficiency
    else:
        fuel = df_timeseries.loc[:, row.fuel_costs_ts]
        eff = df_timeseries.loc[:, row.efficiency_ts]

    p_max_pu = resolve_attribute(row.p_max_pu, row.p_max_pu_ts, df_timeseries)
    p_min_pu = resolve_attribute(row.p_min_pu, row.p_min_pu_ts, df_timeseries)

    return fuel, eff, p_max_pu, p_min_pu

def add_seasonal_effs(index, row, df_timeseries):
    """
    Resolve seasonal efficiencies for a component based on its attributes.

    Args:
        index (Any): Index of the component.
        row (pd.Series): Row from the DataFrame containing component attributes.
        df_timeseries (pd.DataFrame): DataFrame containing timeseries data.

    Returns:
        effs (Union[pd.Series, float]): Seasonal efficiency timeseries or constant value.
    """

    return resolve_attribute(row.standing_loss, row.standing_loss_ts, df_timeseries)

# -------------------------
# Annuity Calculation Wrapper
# -------------------------
def calculate_annuity(filepath, file, INTEREST):
    """
    Calculate the annuity for a given component type based on its investment and lifetime.

    Args:
        filepath (str): Path to the CSV file containing component data.
        file (str): Name of the file used to determine the component type.
        INTEREST (float): Weighted average cost of capital (WACC) as a decimal.

    Returns:
        df (pd.DataFrame): DataFrame containing component data with calculated capital costs.
    """

    df = pd.read_csv(filepath, index_col=0)
    component_type = file.split('.')[0]

    if component_type in ['generators', 'storage_units', 'links', 'stores']:
        df['capital_cost'] = annuity(df['investment'], df['lifetime'], INTEREST) + df['FOM']

    return df

# -------------------------
# Constraint Construction
# -------------------------
def add_area_constraint(network, snapshots, df_generators, df_links, df_stores, df_storageunits):
    """
    Add area constraints based on the maximum area allowed for each component type.

    Args:
        network (pypsa.Network): PyPSA network object.
        snapshots (List[Any]): List of time snapshots.
        df_generators (pd.DataFrame): DataFrame containing generator data.
        df_links (pd.DataFrame): DataFrame containing link data.
        df_stores (pd.DataFrame): DataFrame containing store data.
        df_storageunits (pd.DataFrame): DataFrame containing storage unit data.

    Returns:
        None: Modifies the network object in place by adding area constraints.
    """

    vars_gen_pnom = get_var(network, "Generator", "p_nom")
    vars_lin_pnom = get_var(network, "Link", "p_nom")
    vars_sto_pnom = get_var(network, "Store", "e_nom")
    vars_storage_pnom = get_var(network, "StorageUnit", "p_nom")

    area_constraints = {}

    def group_components(df, comp_type):
        for _, row in df.iterrows():
            if not pd.isna(row.p_nom_max_area):
                max_area = row.p_nom_max_area
                factor = row.factor
                name = row.name
                area_constraints.setdefault(max_area, []).append((name, factor, comp_type))

    group_components(df_generators, "gen")
    group_components(df_links, "lin")
    group_components(df_stores, "sto")
    group_components(df_storageunits, "storage")

    for max_area, components in area_constraints.items():
        total_area_expr = sum(
            vars_gen_pnom[gen] * factor for gen, factor, comp in components if comp == "gen"
        ) + sum(
            vars_lin_pnom[lin] * factor for lin, factor, comp in components if comp == "lin"
        ) + sum(
            vars_sto_pnom[sto] * factor for sto, factor, comp in components if comp == "sto"
        ) + sum(
            vars_storage_pnom[sto] * factor for sto, factor, comp in components if comp == "storage"
        )

        define_constraints(network, total_area_expr, "<=", max_area, f"area_constraint_p_nom_max_area_{max_area}")
    
# -------------------------
# Store Minimum Capacity Enforcement
# -------------------------
def enforce_min_store_if_built(n, snapshots, df_stores):
    """
    Enforce minimum capacity for stores that are extendable and built.

    Args:
        n (pypsa.Network): PyPSA network object.
        snapshots (List[Any]): List of time snapshots.
        df_stores (pd.DataFrame): DataFrame containing store data with minimum capacity attributes.

    Returns:
        None: Modifies the network object in place by adding constraints.
    """

    stores = n.stores[n.stores.e_nom_extendable].index

    if stores.empty:
            return
            
    e_nom = n.model.variables["Store-e_nom"]

    min_caps = df_stores.loc[stores, "e_nom_min_cap"].to_dict()
    min_caps = {store: cap for store, cap in min_caps.items() if not pd.isna(cap)}
    
    big_M = 1e9

    for store in stores:
        if store not in min_caps:
            continue

        # Add binary variables for store build decision
        build_var = n.model.add_variables(name=f"store_build_{store}", binary=True)

        # Access as expressions
        e = e_nom[store]
        b = build_var[store]

        # Constraint 1: e_nom - big_M * build <= 0
        c_max = e - big_M * b <= 0
        n.model.add_constraints(c_max, name=f"store_big_M_upper_{store}")

        # Constraint 2: min_cap * build - e_nom <= 0  ->  e_nom >= min_cap * build
        c_min = min_caps[store] * b - e <= 0
        n.model.add_constraints(c_min, name=f"store_min_if_built_{store}")

# -------------------------
# Link Minimum Capacity Enforcement
# -------------------------
def enforce_min_link_if_built(n, snapshots, df_links):
    """
    Enforce minimum capacity for links that are extendable and built.

    Args:
        n (pypsa.Network): PyPSA network object.
        snapshots (List[Any]): List of time snapshots.
        df_links (pd.DataFrame): DataFrame containing link data with minimum capacity attributes.

    Returns:
        None: Modifies the network object in place by adding constraints.
    """

    links = n.links[n.links.p_nom_extendable].index

    links = [ln for ln in links if not ln.startswith("heating_grid_")]

    p_nom = n.model.variables["Link-p_nom"]

    min_caps = df_links.loc[links, "p_nom_min_cap"].to_dict()
    min_caps = {link: cap for link, cap in min_caps.items() if not pd.isna(cap)}
    
    big_M = 1e9

    for link in links:
        if link not in min_caps:
            continue
        
        # Add binary variables for link build decision
        build_var = n.model.add_variables(name=f"link_build_{link}", binary=True)

        # Access as expressions
        p = p_nom[link]
        b = build_var[link]

        # Constraint 1: e_nom - big_M * build <= 0
        c_max = p - big_M * b <= 0
        n.model.add_constraints(c_max, name=f"link_big_M_upper_{link}")

        # Constraint 2: min_cap * build - e_nom <= 0  ->  e_nom >= min_cap * build
        c_min = min_caps[link] * b - p <= 0
        n.model.add_constraints(c_min, name=f"link_min_if_built_{link}")

# -------------------------
# Piecewise DHN Modeling
# -------------------------
def add_piecewise_dhn_links(network, basepath, REGION, lifetime_dhn, INTEREST):
    """
    Adds heat links with dynamically calculated efficiencies based on normalized heat demand.

    Args:
        network (pypsa.Network): PyPSA network object.
        basepath (str): Base path for the project directory.
        REGION (str): Name of the region (used to find the CSV).
    """

    # Load CSV with loss and capacity info
    df_loss = pd.read_csv(os.path.join(basepath, 'model', REGION, "costs_func_heat_grid.csv"))
    capacity = df_loss['capacity'].values  # MW
    losses = df_loss['efficiency'].values  # Total loss (1 - eff)
    costs = df_loss['cost'].values

    # Load and normalize heat demand
    heat_demand = network.loads_t["p_set"].loc[:, "heat demand"]
    heat_demand_normalized = heat_demand / heat_demand.max()

    T = len(heat_demand)
    N = len(capacity)

    # Broadcast: demand_scaled = normalized_demand * each capacity
    demand_scaled = heat_demand_normalized.values * capacity[:, np.newaxis]  # shape (N, T)
    loss_broadcast = losses[:, np.newaxis]  # shape (N, 1)

    # Avoid division by zero
    demand_scaled[demand_scaled == 0] = np.nan

    # Compute instantaneous efficiency
    efficiency = 1 - (loss_broadcast / demand_scaled)  # shape (N, T)

    # Store results in DataFrame
    efficiency_df = pd.DataFrame(
        efficiency.T,
        index=heat_demand.index,
        columns=[f"{int(c)} MW" for c in capacity]
    )

    # Save efficiency DataFrame to CSV
    efficiency_df.to_csv(os.path.join(basepath, 'results', REGION, 'efficiency_heat_links.csv'))

    # Add links with average efficiency per link (fallback to mean ignoring NaNs)
    added_links = []
    for i in range(N):
        eff_series = efficiency[i, :]  # shape (T,)
        
        cap = capacity[i]
        cost = costs[i]
        annualized_costs = [annuity(cost, lifetime_dhn, INTEREST)]

        cap_min = capacity[i-1] if i > 0 else 0  # Minimum capacity for the first link is 0

        link_name = f"heating_grid_{int(cap)}MW"

        network.add(
            "Link",
            name=link_name,
            bus0='district heat',
            bus1='heat',
            p_nom_max=cap,
            p_nom_extendable=True,
            efficiency=eff_series,
            capital_cost=annualized_costs/cap,  # Cost per MW
        )

        added_links.append(link_name)

# -------------------------
# Minimum DHN Capacity and Exclusivity Constraints
# -------------------------
def enforce_min_heat_link_if_built(n, snapshots, basepath, REGION):
    """
    Enforce minimum capacity for extendable heat grid links if built,
    based on previous capacity tier from df_loss['capacity'].
    Also enforce that only one link can be built at a time.

    Args:
        n (pypsa.Network): PyPSA network object.
        snapshots (list): List of snapshots.
        df_loss (pd.DataFrame): DataFrame with 'capacity' column, sorted by increasing capacity.

    Returns:
        None
    """

    # Filter extendable heat grid links
    links = n.links[n.links.p_nom_extendable].index
    links = [ln for ln in links if ln.startswith("heating_grid_")]
    
    if not links:
        return

    model = n.model
    p_nom = model.variables["Link-p_nom"]

    # Prepare capacity tier lookup (from df_loss)
    df_loss = pd.read_csv(os.path.join(basepath, 'model', REGION, "costs_func_heat_grid.csv"))
    capacity_list = df_loss['capacity'].sort_values().tolist()
    min_caps = {}

    for i in range(0, len(capacity_list)):
        curr = int(capacity_list[i])
        if i == 0:
            # First capacity tier has no previous tier, set min capacity to 0
            prev = 0
        else:
            prev = capacity_list[i - 1]
        link_name = f"heating_grid_{curr}MW"
        min_caps[link_name] = prev

    # Big-M value, should be larger than max expected capacity
    big_M = 1e6

    build_vars = {}  # Store all binaries here

    for link in links:
        if link not in min_caps:
            continue

        # Add binary build variable and store it
        b = model.add_variables(name=f"heat_link_build_{link}", binary=True)
        build_vars[link] = b[link]  # store the scalar variable directly

        p = p_nom[link]
        min_cap = min_caps[link]

        # Constraint 1: p_nom <= big_M * binary
        c_max = p - big_M * build_vars[link] <= 0
        model.add_constraints(c_max, name=f"heat_link_big_M_upper_{link}")

        # Constraint 2: p_nom >= min_cap * binary
        c_min = min_cap * build_vars[link] - p <= 0
        model.add_constraints(c_min, name=f"heat_link_min_if_built_{link}")

    # Add exclusivity constraint: sum of binaries ≤ 1 (only one link can be built)
    exclusivity = sum(build_vars[link] for link in build_vars) <= 1

    model.add_constraints(exclusivity, name="only_one_heat_link_built")
