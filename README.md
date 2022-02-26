# FYP Project C035

## Summary
This project aims to study and model the ferry service operations optimisation problem as a Capacitated Vehicle Routing Problem with Backhauls and Time Windows (VRPBTW).

The objective of this repository is to develop an optimal scheduling system that can generate an optimal set of routes and schedules that best maximises the ability of ferry service operations to meet its daily demands. The following methods were employed to tackle this optimisation problem.

#### 1. Exact methods - Exhaustive search, Linear Programming Model
#### 2. Heuristic methods - Genetic Algorithm

All the above methods were implemented using Python 3.7.
The algorithms were run and tested on a Lenovo YOGA S740 with am Intel i7 CPU and 16 GB RAM.

## Installation

#### 1. Clone this repository in your working directory

```bash
git clone https://github.com/chensxb97/ferryServiceVRP.git
```

#### 2. Install CPLEX using instructions from the link below

https://github.com/IBMPredictiveAnalytics/Simple_Linear_Programming_with_CPLEX/blob/master/cplex_instruction.md


#### 3. Install dependencies

```bash
pip install -r requirements.txt 
```

## Usage

#### 1. Setting the environment variable PYTHONPATH

Change line 2 for the following python scripts to your CPLEX directory: 'yourCplexhome/python/VERSION/PLATFORM'

- exhaustiveSearch.py
- lpModel.py
- schedule.py

#### 2. Test python scripts on a sample dataset: LT1.csv, by default. 

Exhaustive search Algorithm
```python
python exhaustiveSearch.py
```
Linear Programming Model
```python
python lpModel.py
```
Genetic Algorithm
```python
python ga.py
```
The input datasets can be found in the folder: /datasets.
The outputs from the above scripts can be found in folder: /outputs.
Output logs from the lpModel.py and ga.py were manually compiled in GA.txt and lpModel.txt in folder: /datasets/logs.
When lpModel.py and ga.py are run, they generate visualisation maps that are automatically saved in folder: /datasets/plots.

#### 3. Run scheduling system, which optimises the sets of routes for the following test case.

Dataset: datasets/order.csv,

Tours: 0900-1130, 1130-1400,

Fleet size: 5

```python
python schedule.py
```

When schedule.py is run, it generates a timetable, schedule.csv, which is saved in folder: /outputs/logs.
It also generates visualisation maps that are saved in folder: /outputs/plots/schedule

## References
1. [Modelling and Analysis of a Vehicle Routing Problem with Time Windows in Freight Delivery (MIP Model)](https://github.com/dungtran209/Modelling-and-Analysis-of-a-Vehicle-Routing-Problem-with-Time-Windows-in-Freight-Delivery/blob/master/algorithm/VRPTW%20MIP%20Model.pdf)
2. [A Python Implementation of a Genetic Algorithm-based Solution to Vehicle Routing Problem with Time Windows](https://github.com/iRB-Lab/py-ga-VRPTW/tree/c8cab16278b05e9bd641e3afe0b55a34c6c67773)




