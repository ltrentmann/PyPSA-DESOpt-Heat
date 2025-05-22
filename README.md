# PyPSA-DESOpt-Heat
PyPSA for District Energy System Optimization with focus on the Heat Sector

## Intro
PyPSA-DESOpt-Heat is a linear programming district energy system optimization model which design the cost-optimal energy system considering distributed and building-specific heat supply.


## Feature overview

* Sector-coupled district energy system optimization
  * Variable efficiency, price and demand timeseries
  * Greenfield and brownfield optimization
* District heating network build-out and building specific heat supply
* Plotting functions included

## Contents

- [PyPSA-DESOpt-Heat](#PyPSA-DESOpt-Heat)
  - [Intro](#Intro)
  - [Feature overview](#Feature-overview)
  - [Contents](#contents)
  - [Description](#description)
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

The cost function for the DHN can be calculated using the **sensitivity mode** of the **Topotherm** model — a **Pyomo-based mixed-integer linear programming (MILP)** tool for district heating network design.

## Install

1. Create and activate enviroment. Example with anaconda:   `conda activate PyPSADESOptHeat`
2. cd to folder with github clone
3. `pip install -e .`

## Solver

A free academic license of gurobi is available and can be installed by following
the documentation [here](https://support.gurobi.com/hc/en-us/articles/360044290292-How-do-I-install-Gurobi-for-Python-).

## Usage

Run the script main.py

## Contribute

Pull requests and any feedback regarding the code are very welcome. For major
changes, please open an issue first to discuss what you would like to change.


## Example

District heating network heat supply: 

![dhn_heat_supply](https://github.com/ltrentmann/PyPSA-DESOpt-Heat/blob/main/results/grch_60-45/district%20heat_grch_60-45_flow.svg "dhn_heat_supply")
