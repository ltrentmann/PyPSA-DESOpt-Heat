import os
import math
import pandas as pd
import copy
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP
from matplotlib.colors import LinearSegmentedColormap

# === Global Variables ===
mycmap = LinearSegmentedColormap.from_list('mycmap', ['#f9ba00', '#c4071b'])
mycmap_dark = LinearSegmentedColormap.from_list('mycmap_dark', ['#0f1b5f', '#c4071b'])

def capacities(lon, lat, capacity=1):
    """Initializes a dataframe for a location in a cutout and returns a 
    dataframe to pass to layout_from_capacity_list.
    
    Args:
        lon (float): longitude of the location
        lat (float): latitude of the location
        capacity (float): capacity of the plant in kW
    Returns:
        df (pd.DataFrame): dataframe with capacities
    """
    df = pd.DataFrame.from_dict({'index': 0, 'x': lon, 'y': lat, 'Capacity': capacity}, orient='index').T
    return df

def create_dir(path):
    """Creates a directory if it does not exist and deletes old results.
    
    Args:
        path (str): path to directory
    """
    # create results directory
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        # delete old results
        for f in os.listdir(path):
            os.remove(os.path.join(path, f))
    return

def merge_dfs(results):
    """Merge all DataFrames in results to a single DataFrame. Drops duplicates
    in a list of column names and renames the rest of the duplicated column
     names to avoid duplicates.
     
     Args:
         results (dict): dictionary with results in DataFrames with the same
             datetime index.
     Returns:
         df (pd.DataFrame): merged DataFrame
    """
    # list of duplicate cols to drop (which are the same values)
    drop_duplicates = ["x", "y", "lat", "lon", "temperature"]
    # dict keys to skip
    skip_items = ["T_network"]

    d = copy.deepcopy(results)
    df = pd.DataFrame()
    for key in d.keys():
        if key in skip_items:
            continue
        # check if any columns are in drop duplicates and df, if yes, drop them
        if type(d[key]) == pd.Series:
            d[key] = pd.DataFrame(d[key])
        
        cols_drop = d[key].columns.isin(drop_duplicates)
        if cols_drop.any() & df.columns.isin(drop_duplicates).any():
            d[key].drop(columns=d[key].columns[cols_drop], inplace=True)

        # rename rest of columns to avoid duplicates
        cols_to_rename = d[key].columns[d[key].columns.isin(df.columns)]
        if cols_to_rename.any():
            # if yes, add a suffix to the column name for which it is true
            for k in cols_to_rename:
                d[key].rename(columns={k: k + '-' + key},
                                    inplace=True)

        # merge the dataframes
        df = pd.concat([df, d[key]], axis=1)

    return df

# === Color Dict ===
def create_color_dict(network):
    """
    Create a dictionary mapping all component names in the network to their carrier colors.

    Parameters
    ----------
    network : pypsa.Network
        The PyPSA network object.

    Returns
    -------
    dict
        A dictionary mapping component names to carrier colors.
    """
    color_dict = {}
    
    # List of components that have a 'carrier' attribute
    components_with_carrier = [
        "generators", "loads", "links", "storage_units", "stores"
    ]

    for comp in components_with_carrier:
        comp_df = getattr(network, comp, None)
        if comp_df is None or 'carrier' not in comp_df.columns:
            continue

        for name, row in comp_df.iterrows():
            carrier = row["carrier"]
            if pd.notna(carrier) and carrier in network.carriers.index:
                color = network.carriers.at[carrier, "color"]
                if pd.notna(color):
                    color_dict[name] = color

    return color_dict

# === Helper Functions ===
def get_bus_flows(network, bus_name):
    flows = pd.DataFrame(index=network.snapshots)

    for gen in network.generators.index:
        if network.generators.at[gen, 'bus'] == bus_name:
            flows[gen] = network.generators_t.p[gen]

    for storage in network.storage_units.index:
        if network.storage_units.at[storage, 'bus'] == bus_name:
            flows[storage] = network.storage_units_t.p_dispatch[storage]

    for link in network.links.index:
        for i, bus_side in enumerate(['bus0', 'bus1', 'bus2']):
            if network.links.at[link, bus_side] == bus_name:
                flows[link] = -getattr(network.links_t, f'p{i}')[link]

    return flows.reindex(network.snapshots, fill_value=0)

def compute_storage_geometry(energy_MWh, temp_h, temp_c, area_m2):
    pressure = 101325
    temp_K = (temp_h + temp_c) / 2 + 273.15
    density = PropsSI("D", "P", pressure, "T", temp_K, "Water")
    heat_capacity = PropsSI("C", "P", pressure, "T", temp_K, "Water")
    
    volume = energy_MWh * 1e9 * 3.6 / ((temp_h - temp_c) * heat_capacity * density)
    height = volume / area_m2
    diameter = (4 * volume / (math.pi * height)) ** 0.5
    return volume, height, diameter

# === Plot Functions ===
def flow_plot(network, bus_name, order, demand, title, folder):
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({'font.size': 10})

    flows = get_bus_flows(network, bus_name)
    flowsstorage = flows.filter(like='TTES charge')
 
    if "geothermal hp" in flows.columns:
        flows["geothermal hp"] = flows["geothermal hp"].abs()

    flows = flows.dropna(axis=1, how='all')
    flows = flows.loc[:, abs(flows).max() > 0.01]
    flowsheating = flows.filter(like='heating grid')
    flows = flows.drop(columns=flowsheating.columns, errors='ignore')
    flows = flows.clip(lower=0)

    flows = flows[[x for x in order if x in flows.columns]]

    # File paths
    basepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    network_folder = os.path.join(basepath, 'results', folder)

    df_gen = pd.read_csv(os.path.join(network_folder, 'generators.csv'), index_col=0)
    df_links = pd.read_csv(os.path.join(network_folder, 'links.csv'), index_col=0)
    df_stores = pd.read_csv(os.path.join(network_folder, 'stores.csv'), index_col=0)

    # Area constraint calc
    area_constraints = {}
    for df, var_type in [(df_gen, "gen"), (df_links, "lin"), (df_stores, "sto")]:
        for idx, row in df.iterrows():
            if not pd.isna(row.p_nom_max_area):
                area_constraints.setdefault(row.p_nom_max_area, []).append((row.name, row.factor, var_type))

    vars_gen = network.generators.p_nom_opt
    vars_links = network.links.p_nom_opt
    vars_stores = network.stores.e_nom_opt

    total_area_expr = sum(
        sum(vars_gen.get(gen, 0) * factor for gen, factor, typ in comps if typ == "gen") +
        sum(vars_links.get(lin, 0) * factor for lin, factor, typ in comps if typ == "lin") +
        sum(vars_stores.get(sto, 0) * factor for sto, factor, typ in comps if typ == "sto")
        for comps in area_constraints.values()
    )

    COLOR_DICT = create_color_dict(network)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={'width_ratios': [1, 1/3]})
    flows.resample('D').mean().plot(ax=ax1, kind='area', color=COLOR_DICT, alpha=0.8, linewidth=0)

    COLOR_DICT = create_color_dict(network)
    if not flowsstorage.empty:
        flowsstorage.resample('D').mean().plot(ax=ax1, color=COLOR_DICT, alpha=0.8)

    if bus_name == "district heat":
        if not flowsheating.empty:
            flowsheating.resample('D').mean().sum(axis=1).abs().plot(ax=ax1, color='black', alpha=0.8)

    ax1.set_ylabel("Energy in MW")
    ax1.set_xlabel("Date")
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=5, frameon=True, framealpha=0.8)
    ax1.set_ylim(round(flowsstorage.resample('D').mean().sum(axis=1).min()), round(flows.resample('D').mean().sum(axis=1).max()))
    ax1.grid(True)

    sum_values = flows.sum()
    wedges, texts, autotexts = ax2.pie(
        sum_values, autopct='%1.1f%%', startangle=90,
        colors=[COLOR_DICT.get(i, "#CCCCCC") for i in sum_values.index],
        wedgeprops={"alpha": 0.8}
    )
    # Normalize values to percentages
    percentages = (sum_values / sum_values.sum()) * 100

    # Loop through each percentage and adjust label position
    for i, pct in enumerate(percentages):
        x, y = autotexts[i].get_position()  # Original position

        if pct < 5:
            # Very small: move out and left
            x *= 2.2  # push left
            y *= 2
        elif 5 <= pct < 7:
            # Small: move out and right
            x *= 2
            y *= 2
        elif pct < 10:
            # Moderate: just move slightly out
            x *= 2.2
            y *= 2

        autotexts[i].set_position((x, y))
        autotexts[i].set_color("black")


    if bus_name == "district heat":
        total_demand = network.loads_t.p_set["heat demand"].sum()
        lcoh = (network.statistics.capex().sum() + network.statistics.opex().sum() + network.model.variables["pw_link_capcost"].solution) / total_demand
        ax2.text(0, -1.5, f'LCOH: {lcoh:.2f} €/MWh', ha='center', va='center')

        dec = network.generators_t.p.loc[:, network.generators.carrier.str.contains('dec')].sum()
        dec_share = dec.sum() / total_demand
        dec_peak = network.generators_t.p.loc[:, network.generators.carrier.str.contains('dec')].max()
        dec_share_peak = dec_peak.sum() / network.loads_t.p_set["heat demand"].max()

        ax2.text(0, -1.75, f'DHN supply: {(1 - dec_share_peak) * 100:.2f}%', ha='center', va='center')
        ax2.text(0, -2, f'Area: {total_area_expr:.2f} m²', ha='center', va='center')

    plt.tight_layout()
    fig.savefig(os.path.join(network_folder, f'{bus_name}_{folder}_flow.svg'))


def plot_storage_energy(network, bus_name, folder, temp_h, temp_c):
    storage_energy = pd.DataFrame(index=network.snapshots)

    for store in network.stores.index:
        if network.stores.at[store, 'bus'] == bus_name:
            storage_energy[store] = network.stores_t.e[store]

    COLOR_DICT = create_color_dict(network)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={'width_ratios': [1, 1/3]})
    storage_energy.resample('D').mean().plot(ax=ax, linewidth=2, color=[COLOR_DICT.get(col, '#CCCCCC') for col in storage_energy.columns])

    ax.set_ylabel("Energy Stored in MWh")
    ax.set_xlabel("Date")
    ax.grid(True)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=4, frameon=True, framealpha=0.8)

    # Area and geometry calc
    basepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    network_folder = os.path.join(basepath, 'results', folder)
    df_stores = pd.read_csv(os.path.join(network_folder, 'stores.csv'), index_col=0)

    vars_sto = network.stores.e_nom_opt
    
    area_constraints = {}

    for sto, row in df_stores.iterrows():  # Use _ for the index since it is reset
        # Ensure p_nom_max_area is defined and valid
        if not pd.isna(row.p_nom_max_area):
            max_area = row.p_nom_max_area
            factor = row.factor
            generator_name = row.name

            # Group generators by their p_nom_max_area value
            if max_area not in area_constraints:
                area_constraints[max_area] = []
            area_constraints[max_area].append((generator_name, factor, "sto"))

    # recalculate the total area for the storage
    total_area_expr = 0
    for max_area, components in area_constraints.items():
        # Create the left-hand side expression by summing the components' weighted p_nom
        total_area_expr = sum(
            vars_sto[sto] * factor
            for sto, factor, comp_type in components if comp_type == "sto"
        )

    TES = network.stores.e_nom_opt.loc[bus_name]
    fluid = "Water"
    pressure = 101325  # Pa
    temperature = (temp_h + temp_c)/2 + 273.15  # 60°C to Kelvin
    density = CP.PropsSI("D", "P", pressure, "T", temperature, fluid)  # Density in kg/m³
    heat_capacity = CP.PropsSI("C", "P", pressure, "T", temperature, fluid)  # J/kgK
    volume = TES * 1000000000 * 3.6 / ((temp_h - temp_c) * heat_capacity * density)
    diameter = (4 * volume / (math.pi * total_area_expr)) ** 0.5
    height = volume / total_area_expr

    ax2.text(0.5, 0.75, f'Volume: {volume:.2f} m³', ha='center')
    ax2.text(0.5, 0.5, f'Height: {height:.2f} m', ha='center')
    ax2.text(0.5, 0.25, f'Area: {total_area_expr:.2f} m²', ha='center')
    ax2.text(0.5, 0.0, f'Diameter: {diameter:.2f} m', ha='center')
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(network_folder, f'{bus_name}_{folder}_storage_energy.svg'))


def create_summary_table(network):
    """
    Generates a summary table with installed capacity, CapEx, and OpEx for each component.
    
    Parameters:
    -----------
    network : pypsa.Network
        The PyPSA network after running network.optimize().
    
    Returns:
    --------
    pd.DataFrame
        A summary table with Installed Capacity (MW), CapEx (€), and OpEx (€) for generators.
    """
    
    summary_data = []

    # --- Generators ---
    for gen_name, gen in network.generators.iterrows():
        if gen.p_nom_extendable:
            installed_capacity = network.generators.at[gen_name, "p_nom_opt"]
        else:
            installed_capacity = gen.p_nom
        capex = gen.capital_cost * installed_capacity
        op_series = network.generators_t.p[gen_name]
        if gen_name in network.generators_t.marginal_cost:
            opex = (abs(network.generators_t.marginal_cost[gen_name] * op_series)).sum()

        summary_data.append({
            "Component": gen_name,
            "Type": "Generator",
            "Carrier": gen.carrier,
            "Installed Capacity (MW)": installed_capacity,
            "Generation (MWh)": abs(op_series.sum()),
            "CapEx (€)": capex,
            "OpEx (€)": opex
        })

    # --- Extendable Storage Units (optional) ---
    for store_name, store in network.storage_units.iterrows():
        if store.p_nom_extendable:
            installed_capacity = network.storage_units.at[store_name, "p_nom_opt"]
        else:
            installed_capacity = store.p_nom

        capex = store.capital_cost * installed_capacity
        op_series = network.storage_units_t.p[store_name]
        if store_name in network.storage_units_t.marginal_cost:
            opex = (abs(network.storage_units_t.marginal_cost[store_name] * op_series)).sum()

        summary_data.append({
            "Component": store_name,
            "Type": "Storage",
            "Carrier": store.carrier,
            "Installed Capacity (MW)": installed_capacity,
            "Generation (MWh)": abs(op_series.sum()),
            "CapEx (€)": capex,
            "OpEx (€)": opex
        })

    # --- Extendable Stores (optional) ---
    for store_name, store in network.stores.iterrows():
        if store.e_nom_extendable:
            installed_capacity = network.stores.at[store_name, "e_nom_opt"]
        else:
            installed_capacity = store.e_nom

        capex = store.capital_cost * installed_capacity
        op_series = network.stores_t.e[store_name]
        if store_name in network.stores_t.marginal_cost:
            opex = (abs(network.stores_t.marginal_cost[store_name] * op_series)).sum() 

        summary_data.append({
            "Component": store_name,
            "Type": "Store",
            "Carrier": store.carrier,
            "Installed Capacity (MW)": installed_capacity,
            "Generation (MWh)": abs(op_series.sum()),
            "CapEx (€)": capex,
            "OpEx (€)": opex
        })

    # --- Links ---
    for link_name, link in network.links.iterrows():
        # Determine installed capacity
        if link.p_nom_extendable:
            p_nom_opt = network.links.at[link_name, "p_nom_opt"]

            # Check if efficiency is a time series
            if link_name in network.links_t.efficiency:
                mean_efficiency = network.links_t.efficiency[link_name].mean()
            else:
                mean_efficiency = link.efficiency

            installed_capacity = p_nom_opt * mean_efficiency
        else:
            installed_capacity = link.p_nom

        capex = link.capital_cost * installed_capacity
        op_series = network.links_t.p1[link_name]
        if link_name in network.links_t.p2:
            op_series2 = network.links_t.p2[link_name]
        if link_name in network.links_t.marginal_cost:
            opex = (abs(network.links_t.marginal_cost[link_name] * op_series)).sum()

        if link_name == "heating grid":
            capex = network.model.variables["pw_link_capcost"].solution.item()

        summary_data.append({
            "Component": link_name,
            "Type": "Link",
            "Carrier": link.carrier,
            "Installed Capacity (MW)": installed_capacity,
            "Generation (MWh)": abs(op_series.sum()),
            "Generation2 (MWh)": abs(op_series2.sum()) if 'op_series2' in locals() else 0,
            "CapEx (€)": capex,
            "OpEx (€)": opex
        })

    
    # Optional: Add lines, transformers, etc.

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values(by="CapEx (€)", ascending=False).reset_index(drop=True)


    return summary_df
