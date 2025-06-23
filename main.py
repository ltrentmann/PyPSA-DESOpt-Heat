# -*- coding: utf-8 -*-
"""
@author: Lennart Trentmann (lennart.trentmann@tum.de); Amedeo Ceruti
(amedeo.ceruti@tum.de)

This file is used to optimize district energy systems with the tool PyPSA-DESOpt-Heat
of the Chair of Energy Systems (TUM) based on PyPSA from TU Berlin.
"""


# -------------------- Imports --------------------
import os
import pandas as pd
import pypsa
import numpy as np
import matplotlib.pyplot as plt
from time import time
import ast
import logging
import warnings
import pypsatopo
from sklearn.isotonic import IsotonicRegression
from pypsa.optimization.compat import define_constraints, get_var, join_exprs, linexpr
from src.model import *  
from src.utils import *

# -------------------- Configurations --------------------
warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.ERROR)
pypsa.optimization.optimize.logger.setLevel(logging.INFO)

# -------------------- Scenario Setup --------------------
scenarios = [
    {"REGION": "region_60-45", "RESULTS": "region_60-45"},
    # Add more scenarios as needed
]

# -------------------- Main Execution --------------------
for scenario in scenarios:
    REGION = scenario["REGION"]
    RESULTS = scenario["RESULTS"]

    print(f"Running scenario: {REGION}")

    # --- Paths ---
    basepath = os.path.dirname(os.path.realpath(__file__))
    network_folder = os.path.join(basepath, 'model', REGION)
    results_folder = os.path.join(basepath, 'results', REGION)
    os.makedirs(results_folder, exist_ok=True)

    # --- Read Parameters ---
    params_path = os.path.join(network_folder, 'params.csv')
    df_params = pd.read_csv(params_path, sep=';', header=None, names=['key', 'value'])
    params = {row['key']: row['value'] for _, row in df_params.iterrows()}

    INTEREST = float(params['INTEREST'])
    CO2_PRICE = float(params['CO2_PRICE'])
    CO2_LIMIT = float(params['CO2_LIMIT'])
    HOURS = int(params['HOURS'])
    TEMP_SUP = float(params['TEMP_SUP'])
    TEMP_RET = float(params['TEMP_RET'])

    # --- CAPEX Calculation ---
    for file in os.listdir(network_folder):
        if file.endswith(".csv"):
            filepath = os.path.join(network_folder, file)
            df = calculate_annuity(filepath, file, INTEREST)
            df.to_csv(filepath)

    # --- Initialize Network ---
    network = pypsa.Network()

    # --- Load Timeseries ---
    df_timeseries = pd.read_csv(os.path.join(network_folder, 'timeseries.csv'), index_col=0, sep=";").iloc[:HOURS, :]
    df_timeseries.index = pd.to_datetime(df_timeseries.index, format='%d.%m.%Y %H:%M')
    df_timeseries["st panels"] /= 700  # Convert Wh/m² to MWh/MW
    network.set_snapshots(df_timeseries.index.values)
    df_timeseries["normed demand"] = df_timeseries["heat demand"] / df_timeseries["heat demand"].max()

    # --- Load and Import Network Components ---
    def load_and_import(file_name, component):
        df = pd.read_csv(os.path.join(network_folder, file_name), index_col=0)
        network.import_components_from_dataframe(df, component)
        return df

    df_buses = load_and_import('buses.csv', 'Bus')
    df_carriers = load_and_import('carriers.csv', 'Carrier')
    df_loads = load_and_import('loads.csv', 'Load')
    df_generators = load_and_import('generators.csv', 'Generator')
    df_links = load_and_import('links.csv', 'Link')
    df_stores = load_and_import('stores.csv', 'Store')
    df_storageunits = load_and_import('storage_units.csv', 'StorageUnit')

    # --- Assign Load Time Series ---
    for load in df_loads.index:
        network.loads_t.p_set[load] = df_timeseries[df_loads.loc[load, 'ts']]

    # --- Generator Parameters ---
    for gen, row in df_generators.iterrows():
        fuel, eff, p_max_pu, p_min_pu = add_effs_and_marginalcosts(gen, row, df_timeseries)
        network.generators_t["marginal_cost"][gen] = (fuel + row.emission * CO2_PRICE) / pd.Series(eff).mean() + row.VOM
        network.generators_t["efficiency"][gen] = eff
        network.generators_t["p_max_pu"][gen] = p_max_pu
        network.generators_t["p_min_pu"][gen] = p_min_pu
        network.generators.loc[gen, "capital_cost"] = row.capital_cost

    # --- Link Parameters ---
    for link, row in df_links.iterrows():
        fuel, eff, p_max_pu, p_min_pu = add_effs_and_marginalcosts(link, row, df_timeseries)
        network.links_t["efficiency"][link] = eff
        network.links_t["p_max_pu"][link] = p_max_pu
        network.links_t["p_min_pu"][link] = p_min_pu
        network.links_t["marginal_cost"][link] = row.VOM if (pd.Series(eff) == 0).all() else row.VOM * pd.Series(eff).mean()
        network.links.loc[link, "capital_cost"] = row.capital_cost * pd.Series(eff).mean()

    # --- Additional Link Efficiencies ---
    eff_path = os.path.join(network_folder, "links-efficiency2.csv")
    if os.path.exists(eff_path):
        efficiency2 = pd.read_csv(eff_path).iloc[:HOURS, :]
        efficiency2.index = df_timeseries.index
        network.import_series_from_dataframe(efficiency2, 'Link', 'efficiency2')

    # --- Storage Parameters ---
    for sto, row in df_stores.iterrows():
        loss = add_seasonal_effs(sto, row, df_timeseries)
        network.stores_t["standing_loss"][sto] = loss
        network.stores.loc[sto, "capital_cost"] = row.capital_cost
        network.stores_t["marginal_cost"][sto] = row.VOM

    for sto, row in df_storageunits.iterrows():
        loss = add_seasonal_effs(sto, row, df_timeseries)
        network.storage_units_t["standing_loss"][sto] = loss
        network.storage_units.loc[sto, "capital_cost"] = row.capital_cost
        network.storage_units_t["marginal_cost"][sto] = row.VOM

    # --- DHN Link Recalculation ---
    lifetime_dhn = network.links.loc["heating grid", "lifetime"]
    network.remove("Link", "heating grid")

    loss, cap = dhn_eff_calc(network, basepath, REGION, RESULTS)

    network.add(
        "Link", "heating grid",
        bus0="district heat",
        bus1="heat",
        efficiency=1 - (loss / (network.loads_t.p_set["heat demand"] / network.loads_t.p_set["heat demand"].max() * cap)),
        lifetime=lifetime_dhn,
        p_nom_extendable=True,
        # marginal_cost=-103.36,
    )

    # --- Constraints and Optimization ---
    def extra_functionalities(network, snapshots):
        add_area_constraint(network, snapshots, df_generators, df_links, df_stores, df_storageunits)
        add_piecewise_cost_link(network, snapshots, basepath, network_folder, INTEREST, RESULTS)
        enforce_min_store_if_built(network, snapshots, df_stores)

    network.add("GlobalConstraint", "co2_limit", sense="<=", constant=CO2_LIMIT)

    network.consistency_check()

    # generate topographical representation of network 'network' in the SVG format
    pypsatopo.generate(network, file_output = "diagrams/network.svg", file_format = "svg")

    start = time()

    """network.optimize.add_load_shedding(
        buses=["district heat"],
        marginal_cost=100,    # High cost to ensure shedding is last resort
        p_nom=10,              # Maximum allowed shedding (MW)
        sign=1,
    )"""

    network.optimize(
        solver_name='gurobi',
        solver_options={"MIPGap": 0.001, "FeasibilityTol": 1e-4},
        extra_functionality=extra_functionalities
    )

    end = time()
    print(f"Elapsed time: {end - start:.5f} seconds")

    # --- Results ---
    print("Objective value:", network.objective)
    print("SUM:", network.statistics.opex().sum() + network.statistics.capex().sum())
    print("DHN:", network.model.variables["pw_link_capcost"].solution.item())
    print("Objective Check:", network.objective - (network.statistics.opex().sum() + network.statistics.capex().sum()) - network.model.variables["pw_link_capcost"].solution.item())

    # --- Export Results ---
    network.export_to_csv_folder(os.path.join("./results", RESULTS))

    # --- Print Summary ---
    print("\n--- Optimization Results Summary ---")
    print("p_nom_opt (Generators):\n", network.generators.p_nom_opt)
    print("p_nom_opt (Links):\n", network.links.p_nom_opt)
    print("e_nom_opt (Stores):\n", network.stores.e_nom_opt)
    print("p_nom_opt (Storage Units):\n", network.storage_units.p_nom_opt)

    print("\nOPEX:\n", network.statistics.opex())
    print("CAPEX:\n", network.statistics.capex())

    summary = create_summary_table(network)
    summary.to_csv(os.path.join(results_folder, "summary.csv"), sep=';', index=False)

    # --- Plots ---
    # Adapt components which should be plotted #
    # Input: - network: PyPSA Network object
    #        - bus: str, type of plot (e.g., "district heat", "heat", "district elec")
    #        - components: list of str, components to plot
    #        - title: str, title of the plot
    #        - ylabel: str, y-axis label
    #        - region: str, region name for saving the plot

    flow_plot(network, "district heat", 
              ["geothermal plant", "geothermal hp", "st panels", "large scale heat pump", "biomethane CHP", 
               "biomethane boiler", "TTES discharge", "PTES discharge", "elec boiler"],
              "", "district heating supply", REGION)

    flow_plot(network, "heat", 
              ["dec ground heat pump", "dec air heat pump", "dec pellet boiler"],
              "", "building integrated supply", REGION)

    flow_plot(network, "district elec", 
              ["battery storage", "pv panels", "electricity grid", "biomethane CHP"],
              "", "electricity supply", REGION)

    plot_storage_energy(network, "PTES", REGION, TEMP_SUP, TEMP_RET)
    plot_storage_energy(network, "TTES", REGION, TEMP_SUP, TEMP_RET)

    # --- End of Scenario ---
    print(f"Scenario '{REGION}' completed.\n")

print("All scenarios completed.")
