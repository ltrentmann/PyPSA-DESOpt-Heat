"""
Module to calculate the district heating gird.

Authors:
- Lennart Trentmann (lennart.trentmann@tum.de)
- Amedeo Ceruti (amedeo.ceruti@tum.de)
- Jerry Lambert (jerry.lambert@tum.de)
"""

import numpy as np
import iapws.iapws97
import pandas as pd

# Keys required in the temperature levels dictionary for district heating network
TKEYS = ['T_suphigh', 'T_suplow', 'T_turnhigh', 'T_turnlow']
CPWATER = 4.182  # Heat capacity of water in kJ/kgK

def feed_line_temperatures(temp_ambient, df_temps):
    """
    Determine the supply temperature of the DHN (district heating network) based on ambient temperature.
    
    Author: Jerry Lambert (jerry.lambert@tum.de)
    
    The df_temps dictionary must contain:
        - T_suphigh: maximum supply temperature [°C]
        - T_suplow: minimum supply temperature [°C]
        - T_turnhigh: ambient temperature at which T_suphigh is used [°C]
        - T_turnlow: ambient temperature at which T_suplow is used [°C]
    
    The function linearly interpolates the supply temperature between these points.

    Args:
        temp_ambient (pd.DataFrame or atlite.Cutout): Ambient temperature in °C
        df_temps (dict): Dictionary containing the required temperature thresholds

    Returns:
        pd.DataFrame: DataFrame with an additional column `T_supply` for the DHN supply temperature
    """
    
    # Validate dictionary keys
    if not all([k in df_temps.keys() for k in TKEYS]):
        raise ValueError(f"temperatures dict must contain keys {TKEYS}")

    Ts = df_temps

    # Convert to DataFrame if necessary
    if not isinstance(temp_ambient, pd.DataFrame):
        dfTamb = temp_ambient.to_dataframe()
    else:
        dfTamb = temp_ambient

    # Apply piecewise linear interpolation logic for T_supply
    dfTamb.loc[dfTamb.temperature < Ts['T_turnhigh'], 'T_supply'] = Ts['T_suphigh']
    
    dfTamb.loc[
        dfTamb.temperature.between(Ts['T_turnhigh'], Ts['T_turnlow'], inclusive='both'),
        'T_supply'
    ] = Ts['T_suphigh'] - (
        (dfTamb.temperature - Ts['T_turnhigh']) /
        (Ts['T_turnlow'] - Ts['T_turnhigh'])
    ) * (Ts['T_suphigh'] - Ts['T_suplow'])
    
    dfTamb.loc[dfTamb.temperature > Ts['T_turnlow'], 'T_supply'] = Ts['T_suplow']

    # Sanity check for missing values
    if dfTamb.T_supply.isna().any():
        raise ValueError("NaNs in T_supply. Check temperatures dict.")

    return dfTamb


def mass_flow(temp_in, temp_out, Q_th):
    """
    Calculates mass flow in kg/s based on heat power in kW and the temperature difference.

    Args:
        temp_in (float or np.array): Inlet temperature in °C
        temp_out (float or np.array): Outlet temperature in °C
        Q_th (float): Heat demand in kW

    Returns:
        float: Mass flow in kg/s
    """
    # Q_th in kW → convert to kJ/h → divide by (cp * ΔT) → convert to kg/s
    m_dot = Q_th / (CPWATER * np.abs(temp_out - temp_in)) * 3600
    return m_dot


def heat_flow(m_dot, temp_in, temp_out, pressure=1.01325):
    """
    Calculates heat flow in kW using IAPWS-97 water properties (enthalpy difference).

    Args:
        m_dot (float or np.array): Mass flow in kg/s
        temp_in (float or np.array): Inlet temperature in °C
        temp_out (float or np.array): Outlet temperature in °C
        pressure (float): Pressure in bar (default 1.01325 bar)

    Returns:
        float: Heat flow in kW
    """
    # Convert pressure to MPa and temperature to Kelvin
    water_in = iapws.iapws97.IAPWS97(P=pressure / 10, T=temp_in + 273.15)
    water_out = iapws.iapws97.IAPWS97(P=pressure / 10, T=temp_out + 273.15)

    # Compute enthalpy difference in kJ/kg → kW = kg/s * kJ/kg
    Q_th = m_dot * (water_out.h - water_in.h)

    return abs(Q_th) / 1e3  # Convert W to kW

