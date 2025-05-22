import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression

import pypsa
from pypsa.descriptors import get_switchable_as_dense as get_as_dense
from pypsa.optimization.compat import define_constraints, get_var
from src.utils import mycmap, mycmap_dark 

# -------------------------
# Utility Functions
# -------------------------
def annuity(capex, lifetime, wacc):
    return capex * (wacc * (1 + wacc)**lifetime) / ((1 + wacc)**lifetime - 1)

def resolve_attribute(constant, timeseries_key, df_timeseries):
    return df_timeseries.loc[:, timeseries_key] if pd.isna(constant) else constant

# -------------------------
# Timeseries Attribute Assignment
# -------------------------
def add_effs_and_marginalcosts(index, row, df_timeseries):
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
    return resolve_attribute(row.standing_loss, row.standing_loss_ts, df_timeseries)

# -------------------------
# Annuity Calculation Wrapper
# -------------------------
def calculate_annuity(filepath, file, INTEREST):
    df = pd.read_csv(filepath, index_col=0)
    component_type = file.split('.')[0]

    if component_type in ['generators', 'storage_units', 'links', 'stores']:
        df['capital_cost'] = annuity(df['investment'], df['lifetime'], INTEREST) + df['FOM']

    return df

# -------------------------
# Constraint Construction
# -------------------------
def add_area_constraint(network, snapshots, df_generators, df_links, df_stores, df_storageunits):
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
# District Heating Efficiency
# -------------------------

def dhn_eff_calc(network, basepath, REGION, RESULTS):
    # Load and normalize heat demand
    heat_demand = network.loads_t["p_set"].loc[:, "heat demand"]
    heat_demand_normalized = heat_demand / heat_demand.max()

    # Load losses and capacity data
    df_loss = pd.read_csv(os.path.join(basepath, 'model', RESULTS, "costs_func_heat_grid.csv"))
    capacity = df_loss['capacity'].values  # MW
    losses = df_loss['efficiency'].values  # Total loss (1 - efficiency)

    # Broadcast demand and losses
    demand_scaled = heat_demand_normalized.values * capacity[:, np.newaxis]  # Shape: (N, T)
    loss_broadcast = losses[:, np.newaxis]  # Shape: (N, 1)

    # Avoid division by zero
    demand_scaled[demand_scaled == 0] = np.nan

    # Compute instantaneous efficiency
    efficiency = 1 - (loss_broadcast / demand_scaled)  # Shape: (N, T)

    # Convert to DataFrame
    efficiency_df = pd.DataFrame(
        efficiency.T,
        index=heat_demand.index,
        columns=[f"{c:.0f} MW" for c in capacity]
    )

    # Daily mean efficiency
    efficiency_daily = efficiency_df.resample("1D").mean()

    # Plot daily efficiency for all capacities using colormap
    plt.figure(figsize=(6, 6))

    n_lines = len(efficiency_daily.columns)
    for i, col in enumerate(efficiency_daily.columns):
        color = mycmap(i / (n_lines - 1))  # Normalized to [0, 1]
        plt.plot(efficiency_daily.index, efficiency_daily[col], label=col, linewidth=1.5, color=color)

    plt.xlabel("Date")
    plt.ylabel("Efficiency (daily mean)")
    plt.title(f"{REGION} – Daily Avg Efficiency per Capacity")
    # Move legend below the plot
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        fontsize=11,
        ncol=4,  # Adjust number of columns depending on number of labels
        frameon=False
    )
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(basepath, "results", RESULTS, f"efficiency_time_series_daily_{REGION}.svg"))
    plt.close()


    # Plot average efficiency across capacities
    efficiency_avg = efficiency_daily.mean(axis=1)

    plt.figure(figsize=(6, 6))
    plt.plot(efficiency_avg.index, efficiency_avg, color=mycmap_dark(0.5), linewidth=2.5, label="Avg Efficiency")
    plt.xlabel("Date")
    plt.ylabel("Avg Efficiency (daily mean)")
    plt.title(f"{REGION} – Average Daily Efficiency")
    plt.grid(True)
    # Move legend below the plot
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        fontsize=11,
        ncol=4,  # Adjust number of columns depending on number of labels
        frameon=False
    )
    plt.tight_layout()
    plt.savefig(os.path.join(basepath, "results", RESULTS, f"average_efficiency_daily_{REGION}.svg"))
    plt.close()

    # Recompute daily effective losses from average efficiency
    avg_capacity = capacity.mean()
    heat_demand_daily = heat_demand_normalized.resample("1D").mean()
    recomputed_loss_ts = (1 - efficiency_avg.values) * (heat_demand_daily.values * avg_capacity)
    recomputed_loss = np.array(recomputed_loss_ts)[0]

    return recomputed_loss, avg_capacity

# -------------------------
# Piecewise Linear Cost Modeling
# -------------------------
def add_piecewise_cost_link(n, snapshots, basepath, network_folder, INTEREST, RESULTS):
    m = n.model
    link_name = "heating grid"

    df = pd.read_csv(os.path.join(basepath, network_folder, "costs_func_heat_grid.csv"))
    breakpoints, total_costs = df['capacity'].values, df['cost'].values

    lifetime = n.links.loc[link_name, "lifetime"]
    annualized_costs = [annuity(cost, lifetime, INTEREST) for cost in total_costs]

    slopes = np.diff(annualized_costs) / np.diff(breakpoints)
    is_convex = np.all(np.diff(slopes) >= -1e-8)

    if not is_convex:
        ir = IsotonicRegression(increasing=True)
        annualized_costs = ir.fit_transform(breakpoints, annualized_costs)

    slopes, intercepts = [], []
    for i in range(len(breakpoints) - 1):
        a = (annualized_costs[i+1] - annualized_costs[i]) / (breakpoints[i+1] - breakpoints[i])
        b = annualized_costs[i] - a * breakpoints[i]
        slopes.append(a)
        intercepts.append(b)

    capcost = m.add_variables(name="pw_link_capcost", lower=0)

    for i, (a, b) in enumerate(zip(slopes, intercepts)):
        m.add_constraints(capcost >= a * m.variables["Link-p_nom"].loc[link_name] + b,
                          name=f"pw_link_epigraph_seg_{i}")

    m.objective += capcost
