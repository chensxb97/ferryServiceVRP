import sys
sys.path.append('C:/Program Files/IBM/ILOG/CPLEX_Studio201/cplex/python/3.7/x64_win64')
sys.path.insert(0,'C:/users/benedict/appdata/local/programs/python/python37/lib/site-packages')

import argparse
import datetime
import docplex.mp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import time as timer

from docplex.mp.model import Model
from lpTools import drawSolution
from utils import Edges, computeDistMatrix, separateTasks

# Big 'M'
M = 10000
Capacity = 14

MapGraph = nx.Graph()
MapGraph.add_weighted_edges_from(Edges)

time_start = timer.time()

################################################################################################
#-------------------------df format-------------------------------------------------------------
# | Order ID |     Request_Type    |    Zone   | Demand |    Start_TW   |   End_TW   |   Port
# | 0        | 0                   | Port Name | 0      |     TourStart     TourEnd  | Port Name
# | N        | 1-pickup 2-delivery | Zone Name | Amount | TourStart <=  x <= TourEnd | Port Name
################################################################################################

# Linear Programming Model formulation
def calculateRoute(numOfCustomers, numOfVehicles, df):
   
    # Initialisation
    mdl = Model('VRP')

    # Enumerator of 1 - N
    C = [i for i in range(1, numOfCustomers + 1)]

    # Enumerator of 0 - N
    Cc = [0] + C

    # Enumerator of 1 - V
    numOfVehicles = [i for i in range(1, numOfVehicles + 1)]

    # Distance matrix
    distMatrix = computeDistMatrix(df, MapGraph)
    velocity = 0.463 #knot

    # Calculate distances and times from distance matrix
    travDist = {(i, j): distMatrix[i][j] for i in Cc for j in Cc}
    travTime = {(i, j): distMatrix[i][j]/velocity for i in Cc for j in Cc}

    # Constants for computing penalty costs
    earlyPenalty = 10
    latePenalty = 10

    # Calculate service times, time windows, pickup and delivery volumes
    p = [0]
    d = [0]
    readyTime = [0]
    servTime = [0]
    dueTime = [0]
    for i in range(1, numOfCustomers+1):
        servTime.append(df.iloc[i, 3]) # Service time of request
        readyTime.append(df.iloc[i,4]) # Start of time window
        dueTime.append(df.iloc[i,5]) # End of time window
        if df.iloc[i,1] == 1: # Pickup
            p.append(df.iloc[i,3])
            d.append(0)
        else: # Delivery
            p.append(0)
            d.append(df.iloc[i,3])

    # Decision variables set
    Load = [(i, v) for i in Cc for v in numOfVehicles]
    X = [(i, j, v) for i in Cc for j in Cc for v in numOfVehicles]
    T = [(i, v) for i in Cc for v in numOfVehicles]

    # Load decision variables
    load = mdl.integer_var_dict(Load, name='load')
    x = mdl.binary_var_dict(X, name='x')
    t = mdl.integer_var_dict(T, name='t')

    # Constraints
    # All launches depart the depot at time = 0
    mdl.add_constraints(t[0, v] == 0 for v in numOfVehicles)

    # All launches must start at the depot
    mdl.add_constraints(mdl.sum(x[0, j, v] for j in Cc) == 1 for v in numOfVehicles)

    # All launches must return to depot
    mdl.add_constraints(mdl.sum(x[i, 0, v] for i in Cc) == 1 for v in numOfVehicles)

    # All nodes must only be visited once by one launch
    mdl.add_constraints(mdl.sum(x[i, j, v] for i in Cc for v in numOfVehicles if j != i) == 1 for j in C)

    # All launches must exit the node they visited, no stopping allowed
    mdl.add_constraints((mdl.sum(x[i, b, v] for i in Cc if i != b) - mdl.sum(x[b, j, v] for j in Cc if b != j)) == 0 for b in C for v in numOfVehicles)

    # Launch's initial load equals to the total demand for delivery in the route
    mdl.add_constraints((load[0, v] == mdl.sum(x[i, j, v]*d[j] for i in Cc for j in C if i != j)) for v in numOfVehicles)

    # Launch's load balance constraint between nodes i and j
    mdl.add_constraints((load[j, v] >= load[i, v] - d[j] + p[j] - M * (1 - x[i, j, v])) for i in Cc for j in C for v in numOfVehicles if i != j)

    # Total load of each launch at any node does not exceed maximum capacity
    mdl.add_constraints(load[j, v] <= Capacity for j in Cc for v in numOfVehicles)

    # Launch's travelling time balance constraint between nodes i and j
    mdl.add_constraints(t[j,v] >= t[i,v] + servTime[i] + travTime[i,j] - M *(1 - x[i,j,v]) for i in Cc for j in C for v in numOfVehicles if i != j)

    # Total tour duration is strictly less than 2.5hrs
    mdl.add_constraints(mdl.sum(x[i, j, v]*travTime[i, j] + x[i, j, v]*servTime[i] for i in Cc for j in C)<=150 for v in numOfVehicles)
    mdl.add_constraints(mdl.sum(x[i, j, v]*travTime[i, j] + x[i, j, v]*servTime[i] for i in C for j in Cc)<=150 for v in numOfVehicles)

    # Total number of nodes served per launch should be less than 5
    mdl.add_constraints(mdl.sum(x[i, j, v] for i in Cc for j in C) <= 5 for v in numOfVehicles)

    # Objective function
    obj_function = mdl.sum(travDist[i, j] * x[i, j, v] for i in Cc for j in Cc for v in numOfVehicles if i!=j) + mdl.sum((servTime[i])*mdl.max(earlyPenalty*(readyTime[i]-t[i,v]),0,latePenalty*(t[i,v]-dueTime[i])) for i in C for v in numOfVehicles)
    
    # Set time limit
    mdl.parameters.timelimit.set(60)

    # Solve objective function
    mdl.minimize(obj_function)
    time_solve = timer.time()
    solution = mdl.solve(log_output=True)
    time_end = timer.time()
    running_time = round(time_end - time_solve, 2)
    elapsed_time = round(time_end - time_start, 2)

    actualVehicle_usage = 0
    if solution != None:
        route = [x[i, j, k] for i in Cc for j in Cc for k in numOfVehicles if x[i, j, k].solution_value == 1]
        set = [[i, j, k] for i in Cc for j in Cc for k in numOfVehicles if x[i, j, k].solution_value == 1]
        for k in numOfVehicles:
            if x[0, 0, k].solution_value == 0:
                actualVehicle_usage+=1
        return route, set, actualVehicle_usage, obj_function.solution_value
    else:
        print('no feasible solution')
        return None, None, None, None

def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--file', metavar='f', default='example', help='file name of the order book that required to be processed')
    argparser.add_argument('--fleetsize', metavar='l', default='5', help='number of launches available')
    argparser.add_argument('--time', metavar = 't', default='540', help='starting time of optimization, stated in minutes; default at 9AM (540)')
    args = argparser.parse_args()
    dirName = os.path.dirname(os.path.abspath('__file__'))
    file = args.file
    fileName = os.path.join(dirName, 'SampleDataset', file + '.csv')
    order_df = pd.read_csv(fileName, encoding='latin1', error_bad_lines=False)
    order_df = order_df.sort_values(by=['Start_TW','End_TW'])
    fleet = int(args.fleetsize)
    
    # Visualisation map
    img = plt.imread("Port_Of_Singapore_Anchorages_Chartlet.png")
    fig, ax = plt.subplots()
    ax.imshow(img)

    # Pre-optimisation step
    df_MSP, fleetsize_MSP, df_West, fleetsize_West = separateTasks(order_df, fleet)

    # Run LP Model
    route1, solutionSet_West, used_fleet_West, cost1 = calculateRoute(len(df_West)-1, 3, df_West)
    route2, solutionSet_MSP, used_fleet_MSP, cost2= calculateRoute(len(df_MSP)-1, 3, df_MSP)

    # Results
    if route1 == None:
        print('No solution found for West')
    else:
        print('Port West')
        print(df_West)
        print('Solution set: ')
        print(solutionSet_West)
        print('Obj function cost: ', cost1)
        drawSolution(solutionSet_West, df_West, ax, 3)

    print('\n')

    if route2 == None:
        print('No solution found for MSP')
    else:
        print('Port MSP')
        print(df_MSP)
        print('Solution set: ')
        print(solutionSet_MSP)
        print('Obj function cost: ', cost2)
        drawSolution(solutionSet_MSP, df_MSP, ax, 3)
    
    plt.show()

    print('end')

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')










