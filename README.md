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
  - [Feature overview](#Feature overview)
  - [Contents](#contents)
  - [Description](#description)
  - [Getting Started](#getting-started)
    - [Requirements](#requirements)
  - [Install](#install)
    - [Anaconda or mamba](#anaconda-or-mamba)
  - [Solver](#solver)
    - [Gurobi](#gurobi)
  - [Usage](#usage)
  - [Contribute](#contribute)
  - [License](#license)

## Description

To run the model, the pypsa componets have to be defined in the csv network files and a cost function of the district heating network build out has to be defined. For the calcuation of the DHN cost function the sensitivity mode of the pyomo-based mixed-integer linear programming district heating
network design model topotherm can be used.

## Getting Started

This repository needs a PC capable to run python and its standard libraries.

### Requirements

* Anaconda, mamba or venv

## Install

Use git to clone this repository into your computer. Then, install topotherm
with a package manager such as Anaconda, or directly with Python.

### Anaconda or mamba

We recommend to install the dependencies with anaconda or mamba:

```mamba
cd PyPSA-DESOpt-Heat
mamba env create -f environment.yml -n PyPSADESOptHeat
mamba activate PyPSADESOptHeat
```

## Solver

### Gurobi

A free academic license is available and can be installed by following
the documentation [here](https://support.gurobi.com/hc/en-us/articles/360044290292-How-do-I-install-Gurobi-for-Python-).

## Usage

Run the script main.py

## Contribute

Pull requests and any feedback regarding the code are very welcome. For major
changes, please open an issue first to discuss what you would like to change.
