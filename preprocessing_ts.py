"""
Run preprocessing script for time series generation of a given location.

@author: Lennart Trentmenn (lennart.trentmann@tum.de)
@author: Amedeo Ceruti (amedeo.ceruti@tum.de)
"""

import pandas as pd
import atlite
import copy
import warnings

from src import utils
from src.calc import solar
from src.calc import district_heating as dh
from src.calc import heat_pump as hp
from src.calc import wind as wind
from src.calc import thermal_energy_storage as tes

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# =============================
# Global Parameters
# =============================

LATITUDE = 48.262  # Latitude of interest 
LONGITUDE = 11.668  # Longitude of interest 
YEAR = '2022'  # Year of interest
RESOLUTION = 'h'  # Resolution: 'h' for hourly or 'd' for daily

# =============================
# Paths and Constants
# =============================

PATHCUTOUT = f'./data/years/region-{YEAR}.nc'
RESULTSPATH = f'./results/preprocessing/region-{YEAR}/'
DATAPATH = './data/'
MODULE = 'era5'

T_SINK_DEC = [55]  # Dict with °C - Decentral heat pump sink temperature 
QUALITYGRADE = 0.43  # Carnot efficiency quality grade for heat pump
CSV = dict(index_col=0, sep=',')


def cutout(lat, lon, filename):
    """
    Creates a cutout for a given location and year using atlite.
    Returns the cutout object, or loads it from file if it already exists.
    """
    c = atlite.Cutout(
        filename,
        module=MODULE,
        x=slice(lon - .25, lon + .25),
        y=slice(lat - 0.25, lat + 0.25),
        time=YEAR,
        dt=RESOLUTION,
        parallel=True,
    )
    # c.prepare(monthly_requests=True)
    return c


def time_series():
    """
    Main function to calculate time series for the location/year.
    Includes PV, wind, solar thermal, irradiation, and COPs.
    Saves all data to CSVs.
    """
    utils.create_dir(RESULTSPATH)
    results = {}

    # Create atlite cutout
    c = cutout(lat=LATITUDE, lon=LONGITUDE, filename=PATHCUTOUT)

    # Set up layout
    capacities = utils.capacities(lon=LONGITUDE, lat=LATITUDE, capacity=1)
    layout = c.layout_from_capacity_list(capacities)

    # Extract nearest grid point data
    ds = c.data.sel({'x': LONGITUDE, 'y': LATITUDE}, method='nearest')

    # PV Time Series
    df_pv = pd.read_csv(DATAPATH + 'pv-orientations.csv', **CSV)
    results['pv'] = solar.pv(c, layout, orientations=df_pv)
    results['pv'].to_csv(RESULTSPATH + "pv.csv")
    print(RESULTSPATH + "pv.csv")

    # Wind Time Series
    df_wind = pd.read_csv(DATAPATH + 'wind.csv', **CSV)
    results['wind'] = wind.wind(c, layout, df_turbines=df_wind)
    results['wind'].to_csv(RESULTSPATH + "wind.csv")
    print(RESULTSPATH + "wind.csv")

    # Irradiation
    results['irradiation'] = solar.irradiation(c, 'latitude_optimal', lat=LATITUDE, lon=LONGITUDE)
    results['irradiation'].to_csv(RESULTSPATH + "irradiation.csv")
    print(RESULTSPATH + "irradiation.csv")

    # Solar Thermal
    df_st = pd.read_csv(DATAPATH + 'solar-thermal.csv', **CSV)
    results['st'] = solar.thermal_collector(
        results['irradiation'],
        lat=LATITUDE, lon=LONGITUDE,
        df_collectors=df_st
    )
    results['st'].to_csv(RESULTSPATH + "st.csv")
    print(RESULTSPATH + "st.csv")

    # Decentral Heat Pump COPs
    cops_dec = {}
    for t_sink in T_SINK_DEC:
        delta_air = abs(ds['temperature'].squeeze().to_series() - (t_sink + 273.15))
        delta_ground = abs(ds['temperature'].squeeze().to_series() - (t_sink + 273.15))

        cops_dec[f'DeltaT-air-sink{t_sink}'] = delta_air
        cops_dec[f'DeltaT-ground-sink{t_sink}'] = delta_ground

        cops_dec[f'COP-dec-air-sink{t_sink}'] = hp.cop_decentral(delta_temp=delta_air, source='air')
        cops_dec[f'COP-dec-ground-sink{t_sink}'] = hp.cop_decentral(delta_temp=delta_ground, source='soil')

        df_cop = pd.DataFrame.from_dict(cops_dec)
        results['cop-decentral'] = df_cop
        df_cop.to_csv(RESULTSPATH + f"cop-dec-sink{t_sink}.csv")
        print(RESULTSPATH + f"cop-sink{t_sink}.csv")
    
    # Standing losses thermal energy storage
    df_sto = pd.read_csv(DATAPATH + 'thermal-energy-storage.csv', **CSV)
    # Create an empty DataFrame to collect all results
    combined_losses = pd.DataFrame()

    for storage in df_sto.index:
        row = df_sto.loc[storage]
        id_ = row['id']
        loss_series = tes.standing_loss(row, results['irradiation']["temperature"]-273.15)
        combined_losses[id_] = loss_series

    combined_losses.to_csv(RESULTSPATH + "standing-losses-all.csv")
    print(RESULTSPATH + "standing-losses-all.csv")

    # Central DH Network and Large Heat Pump COPs
    df_nets = pd.read_csv(DATAPATH + 'heat-sources.csv', skiprows=[1], header=0, **CSV)
    print(df_nets)

    for i in df_nets.index:
        id_ = df_nets.loc[i, "id"]

        results[f'T_network-{id_}'] = dh.feed_line_temperatures(
            ds['temperature'] - 273.15,
            df_temps=df_nets.loc[i, :]
        )
        results[f'T_network-{id_}'].to_csv(RESULTSPATH + f"T_network-{id_}.csv")

        results[f'cop-jesper-{id_}'] = hp.cop_jesper(
            df_sink=copy.deepcopy(results[f'T_network-{id_}']),
            temp_source=df_nets.loc[i, 'T_source'],
            df_model=pd.read_csv("./data/jesper_model.csv", header=0, sep=",", index_col=1)
        )

        results[f'cop-jesper-{id_}'].to_csv(RESULTSPATH + f"cop-central-jesper-{id_}.csv")

    print(RESULTSPATH + "T_network.csv")
    print(RESULTSPATH + "cop-central.csv")
    return results


def merge_dfs(results):
    """
    Merge all DataFrames in the results dict into a single DataFrame.
    Avoids column name conflicts and drops duplicate metadata columns.
    """
    drop_duplicates = ["x", "y", "lat", "lon", "temperature"]
    skip_items = ["T_network"]

    d = copy.deepcopy(results)
    df = pd.DataFrame()

    for key in d.keys():
        if key in skip_items:
            continue

        cols_drop = d[key].columns.isin(drop_duplicates)
        if cols_drop.any() & df.columns.isin(drop_duplicates).any():
            d[key].drop(columns=d[key].columns[cols_drop], inplace=True)

        cols_to_rename = d[key].columns[d[key].columns.isin(df.columns)]
        if cols_to_rename.any():
            for k in cols_to_rename:
                d[key].rename(columns={k: k + '-' + key}, inplace=True)

        df = pd.concat([df, d[key]], axis=1)
        df.loc['description', d[key].columns.values] = key

    return df


def potentials():
    """
    Main function to calculate thermal energy potentials.
    Currently includes max heat flow and maximum HP output.
    """
    pots = {}
    pots['Q_source'] = dh.heat_flow(m_dot=130, temp_in=57, temp_out=35, pressure=1.01325)
    pots['Q_max_HP'] = hp.max_heat_power(Q_source=pots['Q_source'], cop=4)

    # Future extensions:
    # pots['heat-pump'] = geothermal_potential(lat, lon)
    return pots


if __name__ == "__main__":
    print('============================')
    print('Performing preprocessing...')
    print('============================\n')

    print('============================')
    print(f"Processing year: {YEAR}")
    print('============================')

    res = time_series()
    df = merge_dfs(res)
    df.to_csv(RESULTSPATH + "all-time-series.csv", sep=";", decimal=",")
    test = potentials()
    print(test)
