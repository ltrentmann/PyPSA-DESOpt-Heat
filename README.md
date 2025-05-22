# PyPSA-DESOpt-Heat
PyPSA for District Energy System Optimization with focus on the Heat Sector

## Intro
PyPSA-DESOpt-Heat is a linear programming district energy system optimization model for designing cost-optimal district energy systems. It focuses on distributed and building-specific heat supply and integrates sector coupling within energy system modeling.


## Feature overview

* Sector-coupled district energy system optimization
  * Variable efficiency, price and demand timeseries
  * Greenfield and brownfield optimization
* District heating network (DHN) build-out and building-specific heat supply
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

The cost function for the district heating network (DHN) can be calculated using the **sensitivity mode** of the **[Topotherm](https://github.com/jylambert/topotherm)** model — a **Pyomo-based mixed-integer linear programming (MILP)** tool for district heating network design.

For more details, visit the [Topotherm GitHub repository](https://github.com/jylambert/topotherm).


For a more detailed description of the PyPSA components, refer to the [PyPSA Documentation](https://pypsa.readthedocs.io/) or consult the following publication:

> T. Brown, J. Hörsch, D. Schlachtberger, *PyPSA: Python for Power System Analysis*, Journal of Open Research Software, 6(1), 2018. DOI: [10.5334/jors.188](https://doi.org/10.5334/jors.188). Available at [arXiv:1707.09913](https://arxiv.org/abs/1707.09913).


## Installation

1. Clone this repository and navigate into it:

    ```bash
    git clone https://github.com/<your-username>/PyPSA-DESOpt-Heat.git
    cd PyPSA-DESOpt-Heat
2. Install the package in editable mode: 

    ```bash
    pip install -e .
3. Create and activate a virtual environment. For example, using Anaconda:

   ```bash
   conda env create -f environment.yml
   conda activate PyPSADESOptHeat
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

![dhn_heat_supply](https://github.com/ltrentmann/PyPSA-DESOpt-Heat/blob/main/results/grch_60-45/district%20heat_grch_60-45_flow.svg "dhn_heat_supply")
