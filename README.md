# PyPSA-DESOpt-Heat
PyPSA for District Energy System Optimization with focus on the Heat Sector

## Intro
PyPSA-DESOpt-Heat is a linear programming district energy system optimization model for designing cost-optimal district energy systems. It focuses on distributed and building-specific heat supply and integrates sector coupling within energy system modeling.


## Feature overview

* Sector-coupled district energy system optimization
  * Variable efficiency, price and demand timeseries
  * Greenfield and brownfield optimization
* Cost-optimal decision between district heating network (DHN) build-out and building-specific heat supply
* Preprocessing functions included for timeseries calcuation of COPs, PV, ST and standing losses of TES

## Contents

- [PyPSA-DESOpt-Heat](#PyPSA-DESOpt-Heat)
  - [Intro](#Intro)
  - [Feature overview](#Feature-overview)
  - [Contents](#contents)
  - [Description](#description)
      - [Components of the District Energy System Model](#description-DES)
      - [Timeseries Preprocessing](#description-Pre)
  - [Install](#install)
  - [Solver](#solver)
  - [Usage](#usage)
  - [Contribute](#contribute)
  - [License](#license)
  - [Example](#Example)

## Description

To run the model, ensure the following:

- All required **PyPSA components** are defined in the CSV network files located in the `model` folder.
- A **cost function** for the district heating network (DHN) build-out is specified in the `cost_func_heat_grid.csv` file.

### Components of the District Energy System Model
The components must be defined as follows:

1) buses:
```
name, carrier
```

2) carriers:
```
name,co2_emissions
```
3) loads:
```
name,bus,ts
```
where ts is the column name in timeseries.csv containing the series of demands in hourly resolution

4) generators:
```
name,bus,carrier,efficiency,build_year,lifetime,p_nom_extendable,p_nom_max,p_max_pu,p_min_pu,min_up_time,min_down_time,up_time_before,down_time_before,investment,fuel_costs,FOM,VOM,fuel_costs_ts,efficiency_ts,emission,p_max_pu_ts,p_min_pu_ts,p_nom_max_area,factor
```

To define the time series XX_ts, indicate the name of the relevant column in the timeseries.csv file, leaving the XX empty. An area constraint can be defined via the p_nom_max_area and factor columns, where the total usable area and the conversion factor in m²/MW are defined.

5) links:
```
name,bus0,bus1,carrier,efficiency,build_year,lifetime,p_nom_extendable,p_nom_max,p_max_pu,p_min_pu,min_up_time,min_down_time,up_time_before,down_time_before,ramp_limit_up,ramp_limit_down,bus2,efficiency2,bus3,efficiency3,investment,fuel_costs,FOM,VOM,efficiency_ts,fuel_costs_ts,p_max_pu_ts,p_min_pu_ts,picewise_costs_func,p_nom_max_area,factor
```

An additional convex cost function can also be defined by specifying the name of the CSV file containing the piecewise data. This file must contain the columns: costs, capacity and optional the efficiency. 

6) storage_units:
```
name,bus,p_nom_extendable,p_nom_max,carrier,build_year,lifetime,cyclic_state_of_charge,max_hours,efficiency_store,efficiency_dispatch,standing_loss_ts,standing_loss,investment,FOM,VOM,p_nom_max_area,factor
```

The standing loss can either be defined as timeseries in standing_loss_ts or as constant value in standing_loss.

7) stores:
```
name,bus,carrier,e_nom_extendable,e_nom_min,e_nom_max,standing_loss,standing_loss_ts,build_year,e_cyclic,lifetime,investment,FOM,VOM,p_nom_max_area,factor,e_nom_min_cap,description,capital_cost
```


The cost function for the district heating network (DHN) can be calculated using the **sensitivity mode** of the **[topotherm](https://github.com/jylambert/topotherm)** model — a **Pyomo-based mixed-integer linear programming (MILP)** tool for district heating network design.

For more details, visit the [topotherm GitHub repository](https://github.com/jylambert/topotherm) or refer to the original research paper:
> Lambert, Jerry and Ceruti, Amedeo and Spliethoff, Hartmut, Benchmark of Mixed-Integer Linear Programming Formulations for District Heating Network Design. Energy, Volume 308, 2024, 132885, ISSN 0360-5442, https://doi.org/10.1016/j.energy.2024.132885

For a more detailed description of the PyPSA components, refer to the [PyPSA Documentation](https://pypsa.readthedocs.io/) or consult the following publication:

> T. Brown, J. Hörsch, D. Schlachtberger, *PyPSA: Python for Power System Analysis*, Journal of Open Research Software, 6(1), 2018. DOI: [10.5334/jors.188](https://doi.org/10.5334/jors.188). Available at [arXiv:1707.09913](https://arxiv.org/abs/1707.09913).

### Timeseries Preprocessing
If generation timeseries of renewables like, PV, ST or COPs and standing loss timeseries of TES are needed run `preprocessing_ts.py`

The required technology data parameters must be specified in the relevant CSV files in the data folder.


## Installation

1. Clone this repository and navigate into it:

    ```bash
    git clone https://github.com/<your-username>/PyPSA-DESOpt-Heat.git
    cd PyPSA-DESOpt-Heat

2. Create and activate a virtual environment for PyPSA-DESOpt-Heat and preprocessing. For example, using Anaconda:

   ```bash
   conda env create -f env_main.yml
   conda activate PyPSA-DESOpt-Heat-main

   conda env create -f env_preprocessing.yml
   conda activate PyPSA-DESOpt-Heat-preprocessing

## Solver

We recommend using gurobi, which offers free academic licenses. For installation instructions, see the [installation guide](https://support.gurobi.com/hc/en-us/articles/360044290292-How-do-I-install-Gurobi-for-Python-).

## Usage

Run the main optimization script:
   ```python main.py```

## Contribute

Contributions are welcome! Please open issues to discuss proposed changes or features before submitting pull requests. This helps ensure alignment with project goals.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Example

Below is an example visualization of district heating network heat supply:

![dhn_heat_supply](https://github.com/ltrentmann/PyPSA-DESOpt-Heat/blob/main/results/region_60-45/district%20heat_region_60-45_flow.svg "dhn_heat_supply")
