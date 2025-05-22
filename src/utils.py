import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP
from matplotlib.colors import LinearSegmentedColormap

# === Global Variables ===
COLOR_DICT = {
    "geothermal plant": "#0f1b5f", "geothermal hp": "#0f1b5f", "st panels": "#d64c13",
    "biomethane CHP": "#007c30", "biomethane CHP generator": "#007c30",
    "biomethane boiler": "#679a1d", "PTES charge": "#f9ba00", "PTES discharge": "#f9ba00",
    "large scale heat pump": "#00778a", "pv panels": "#f9ba00", "battery storage": "#ffdc00",
    "electricity grid": "#CCCCCC", "TTES": "#c4071b", "TTES charge": "#f9ba00", "TTES discharge": "#f9ba00",
    "dec air heat pump": "#005293", "dec ground heat pump": "#0f1b5f", "dec pellet boiler": "#007c30",
    "TTES storage": "#c4071b", "elec boiler": "#CCCCCC"
}

mycmap = LinearSegmentedColormap.from_list('mycmap', ['#f9ba00', '#c4071b'])
mycmap_dark = LinearSegmentedColormap.from_list('mycmap_dark', ['#0f1b5f', '#c4071b'])

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
    flowsstorage = flows.filter(like='PTES charge')
    if "geothermal hp" in flows.columns:
        flows["geothermal hp"] = flows["geothermal hp"].abs()

    flows = flows.dropna(axis=1, how='all')
    flows = flows.loc[:, flows.abs().max() > 0.01]
    flowsheating = flows.filter(like='heating grid')
    flows = flows.drop(columns=flowsheating.columns, errors='ignore').clip(lower=0)

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

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={'width_ratios': [1, 1/3]})
    flows.resample('D').mean().plot(ax=ax1, kind='area', color=COLOR_DICT, alpha=0.8, linewidth=0)

    if not flowsstorage.empty:
        flowsstorage.resample('D').mean().plot(ax=ax1, color=COLOR_DICT, alpha=0.8)
    if not flowsheating.empty:
        flowsheating.resample('D').mean().sum(axis=1).abs().plot(ax=ax1, color='black', alpha=0.8)

    ax1.set_ylabel("Energy in MW")
    ax1.set_xlabel("Date")
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=5, frameon=True, framealpha=0.8)
    ax1.set_ylim(0, round(flows.resample('D').mean().sum(axis=1).max()))
    ax1.grid(True)

    sum_values = flows.sum()
    wedges, texts, autotexts = ax2.pie(
        sum_values, autopct='%1.1f%%', startangle=90,
        colors=[COLOR_DICT.get(i, "#CCCCCC") for i in sum_values.index],
        wedgeprops={"alpha": 0.8}
    )

    percentages = sum_values / sum_values.sum() * 100
    for i, (autotext, pct) in enumerate(zip(autotexts, percentages)):
        x, y = autotext.get_position()
        scale = 2 if pct < 10 else 1
        autotext.set_position((x * scale, y * scale))
        if pct < 4:
            # Use the index i instead of lookup by text
            shift = -0.2 if i % 2 == 0 else 0.2
            autotext.set_position((autotext.get_position()[0] + shift, y))
        autotext.set_color("black")

    capcost = network.objective - network.statistics.capex().sum() - network.statistics.opex().sum()
    print('Capcost:', capcost)

    if bus_name == "district heat":
        total_demand = network.loads_t.p_set["heat demand"].sum()
        lcoh = (network.statistics.capex().sum() + network.statistics.opex().sum() + capcost) / total_demand
        ax2.text(0, -1.5, f'LCOH: {lcoh:.2f} €/MWh', ha='center', va='center')

        dec = network.generators_t.p.loc[:, network.generators.carrier.str.contains('dec')].sum()
        dec_share = dec.sum() / total_demand
        dec_peak = network.generators_t.p.loc[:, network.generators.carrier.str.contains('dec')].max()
        dec_share_peak = dec_peak.sum() / network.loads_t.p_set["heat demand"].max()

        ax2.text(0, -1.75, f'DHN supply: {(1 - dec_share_peak) * 100:.2f}%', ha='center', va='center')
        ax2.text(0, -2, f'Area: {total_area_expr:.2f} m²', ha='center', va='center')

        dec_capex = network.statistics.capex().reset_index()
        dec_opex = network.statistics.opex().reset_index()
        dec_capex_sum = dec_capex[dec_capex['carrier'].str.contains('Decentral', na=False)][0].sum()
        dec_opex_sum = dec_opex[dec_opex['carrier'].str.contains('Decentral', na=False)][0].sum()

        lcoh_dhn = (network.statistics.capex().sum() + network.statistics.opex().sum() + capcost - dec_opex_sum - dec_capex_sum) / (total_demand * (1 - dec_share))
        ax2.text(0, -2.25, f'LCOH_dhn: {lcoh_dhn:.2f} €/MWh', ha='center', va='center')

    plt.tight_layout()
    fig.savefig(os.path.join(network_folder, f'{bus_name}_{folder}_flow.svg'))


def plot_storage_energy(network, bus_name, folder, temp_h, temp_c):
    storage_energy = pd.DataFrame(index=network.snapshots)

    for store in network.stores.index:
        if network.stores.at[store, 'bus'] == bus_name:
            storage_energy[store] = network.stores_t.e[store]

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
    print('Density: ', density, 'kg/m³')
    heat_capacity = CP.PropsSI("C", "P", pressure, "T", temperature, fluid)  # J/kgK
    print('Heat capacity: ', heat_capacity, 'J/kgK')
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

