""""
Wind calculations for the energy system model pre-processing.

Authors:
- Lennart Trentmann (lennart.trentmann@tum.de)
- Amedeo Ceruti (amedeo.ceruti@tum.de)
"""

import pandas as pd

def wind(cutout, layout, df_turbines):
    """
    Calculates wind power output for given turbine configurations.

    Args:
        cutout (atlite.Cutout): Atlite cutout containing weather data.
        layout (atlite.Layout): Spatial layout of wind turbines.
        df_turbines (pd.DataFrame): DataFrame with turbine IDs supported by atlite.
            Use `atlite.resource.get_oedb_windturbineconfig()` to retrieve valid IDs.

    Returns:
        pd.DataFrame: Wind power time series per turbine type with additional
                      columns for total and mean output.
    """
    wind_time_series = {}  # Dictionary to store individual turbine time series

    # Loop through each turbine configuration and calculate time series
    for _, v in df_turbines.iterrows():
        wind_time_series[v.id] = cutout.wind(
            turbine=v.id,
            show_progress=False,
            layout=layout
        ).squeeze().to_series()

    # Create DataFrame from dictionary of time series
    df = pd.DataFrame.from_dict(wind_time_series)

    # Add aggregate metrics
    df['total'] = df.sum(axis=1)  # Total power across all turbines
    df['mean'] = df.mean(axis=1)  # Mean power across all turbines

    return df
