# FYP Project C035

## Summary
This project aims to study and model the ferry service operations optimisation problem as a Capacitated Vehicle Routing Problem with Backhauls and Time Windows (VRPBTW).

The objective of this repository is to develop an optimal scheduling system that can generate an optimal set of routes and schedules that best maximises the ability of ferry service operations to meet its daily demands. The following methods were employed to tackle this optimisation problem.

#### 1. Exact methods - Exhaustive search, Classic Linear Programming Model
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

Exhaustive search
```python
python exhaustiveSearch.py
```

Classic Linear Programming Model
```python
python lpModel.py
```

Genetic Algorithm
```python
python ga.py
```

#### 3. Run scheduling system, which optimises the sets of routes for the following test case.

Dataset: SampleDataset/order.csv,

Tours: 0900-1130, 1130-1400,

Fleet size: 5

```python
python schedule.py
```






