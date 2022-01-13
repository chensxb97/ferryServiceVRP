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
from lpTools import printSolution
from utils import Color, Edges, Locations, computeDistMatrix, separateTasks
# from scipy.spatial import distance_matrix

# the big 'M'
M = 10000
Capacity = 14
MapGraph = nx.Graph()
MapGraph.add_weighted_edges_from(Edges)

time_start = timer.time()

###########################################################
#-------------------------df format------------------------
# | Order ID |     Request_Type    |    Zone   | Demand |
# | 0        | 0                   | Port Name | 0      |
# | N        | 1-pickup 2-delivery | Zone Name | Amount |
###########################################################
def calculateRoute(numOfCustomers, numOfVehicles, df):
    print(df)

    mdl = Model('VRP')
    # create enumarator of 1 - N
    C = [i for i in range(1, numOfCustomers + 1)]
    # create enumarator of 0 - N
    Cc = [0] + C
    # create enumarator of 1 - V
    numOfVehicles = [i for i in range(1, numOfVehicles + 1)]

    # get distance matrix
    distMatrix = computeDistMatrix(df, MapGraph)
    velocity = 0.463 #knot

    # calculate distance and time
    cost = {(i, j): distMatrix[i][j] for i in Cc for j in Cc}
    time = {(i, j): distMatrix[i][j]/velocity for i in Cc for j in Cc}

    # calculate service times, time windows, pickup and delivery volume
    servTime = [0]
    timeWindows = [(None,None)]
    p = [0]
    d = [0]
    for i in range(1, numOfCustomers+1):
        servTime.append(df.iloc[i, 3])
        timeWindows.append((df.iloc[i,4],df.iloc[i,5]))
        if df.iloc[i,1] == 2:
            d.append(df.iloc[i,3])
            p.append(0)
        else:
            d.append(0)
            p.append(df.iloc[i,3])

    # decision variables set
    Load = [(i, v) for i in Cc for v in numOfVehicles]
    Index = [i for i in Cc]
    X = [(i, j, v) for i in Cc for j in Cc for v in numOfVehicles]
    T = [(i, v) for i in Cc for v in numOfVehicles] # NEW

    # load decision variables
    x = mdl.binary_var_dict(X, name='x')
    arrTime = mdl.binary_var_dict(T, name='t') # NEW
    load = mdl.integer_var_dict(Load, name='load')
    # index = mdl.integer_var_dict(Index, name='index')

    # defining constraints

    # Starting Time constraint(NEW)
    mdl.add_constraints(arrTime[0, v] == 0 for v in numOfVehicles)

    # Travelling time + Service Time equation(NEW)
    mdl.add_constraints(arrTime[j, v] == x[i, j, v]*(arrTime(i,v) + servTime[i] + time[i,j]) for i in Cc for j in Cc for v in numOfVehicles if i!=j)

    # All vehicles will start at the depot
    mdl.add_constraints(mdl.sum(x[0, j, v] for j in Cc) == 1 for v in numOfVehicles)

    # All vehicles will return to depot
    mdl.add_constraints(mdl.sum(x[i, 0, v] for i in Cc) == 1 for v in numOfVehicles)

    # All nodes will only be visited once by one vehicle
    mdl.add_constraints(mdl.sum(x[i, j, v] for i in Cc for v in numOfVehicles if j != i) == 1 for j in C)

    # Vehicle must exit the node it visited, no stopping at nodes
    mdl.add_constraints((mdl.sum(x[i, b, v] for i in Cc if i != b) - mdl.sum(x[b, j, v] for j in Cc if b != j)) == 0 for b in C for v in numOfVehicles)

    # Subtour Elimination constraint (Not necessary)
    # mdl.add_constraint(index[0] == 0)
    # mdl.add_constraints(1 <= index[i] for i in C)
    # mdl.add_constraints(numOfCustomers + 1 >= index[i] for i in C)
    # mdl.add_constraints(index[i]-index[j]+1<=(numOfCustomers)*(1-x[i, j, v]) for i in C for j in C for v in numOfVehicles if i != j)

    # Vehicle initial load is the total demand for delivery in the route
    mdl.add_constraints((load[0, v] == mdl.sum(x[i, j, v]*d[j] for j in C for i in Cc if i != j)) for v in numOfVehicles)
    mdl.add_constraints((load[j, v] >= load[i, v] - d[j] + p[j] - M * (1 - x[i, j, v])) for i in Cc for j in C for v in numOfVehicles if i != j)

    # Total load does not exceed vehicle capacity
    mdl.add_constraints(load[j, v] <= Capacity for j in Cc for v in numOfVehicles)

    # Total tour duration is strictly less than 2.5hrs(NEW)
    mdl.add_constraints(mdl.sum(x[i, j, v]*time[i, j] + x[i, j, v]*servTime[i] for i in Cc for j in C)<=150 for v in numOfVehicles)
    mdl.add_constraints(mdl.sum(x[i, j, v]*time[i, j] + x[i, j, v]*servTime[i] for i in C for j in Cc)<=150 for v in numOfVehicles)

    # REMOVE (Not necessary)
    # mdl.add_constraints(mdl.sum(x[i, j, v] for i in Cc for j in C) <= 5 for v in numOfVehicles)

# Objective Function
# Minimize the total loss of revenue + cost (CHANGE)
    obj_function = mdl.sum(cost[i, j] * x[i, j, v] for i in Cc for j in Cc for v in numOfVehicles if i!=j)

    # Set time limit
    mdl.parameters.timelimit.set(60)

    # Solve
    mdl.minimize(obj_function)
    time_solve = timer.time()
    solution = mdl.solve(log_output=True)
    time_end = timer.time()
    # print(solution)
    running_time = round(time_end - time_solve, 2)
    elapsed_time = round(time_end - time_start, 2)

    actualVehicle_usage = 0
    if solution != None:
        route = [x[i, j, k] for i in Cc for j in Cc for k in numOfVehicles if x[i, j, k].solution_value == 1]
        set = [[i, j, k] for i in Cc for j in Cc for k in numOfVehicles if x[i, j, k].solution_value == 1]
        for k in numOfVehicles:
            if x[0, 0, k].solution_value == 0:
                actualVehicle_usage+=1
        print(set)
        print(obj_function.solution_value)
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
    # dirName = os.path.dirname(os.path.realpath('__file__'))
    dirName = os.path.dirname(os.path.abspath('__file__')) # Path to directory
    file = args.file
    fileName = os.path.join(dirName, 'SampleDataset', file + '.csv')
    order_df = pd.read_csv(fileName, encoding='latin1', error_bad_lines=False)
    fleet = int(args.fleetsize)
    # img = plt.imread("Singapore-Anchorages-Chart.png")
    img = plt.imread("Port_Of_Singapore_Anchorages_Chartlet.png")
    fig, ax = plt.subplots()
    ax.imshow(img)

    df_MSP, fleetsize_MSP, df_West, fleetsize_West = separateTasks(order_df, fleet)
    
    route1, solutionSet_West, used_fleet_West, cost1 = calculateRoute(len(df_West)-1, 3, df_West) # CHANGE: calculateRoute
    printSolution(solutionSet_West, df_West, ax, 3)
    route2, solutionSet_MSP, used_fleet_MSP, cost2= calculateRoute(len(df_MSP)-1, 3, df_MSP)
    printSolution(solutionSet_MSP, df_MSP, ax, 3)

    print(cost1)
    print(cost2)

    plt.show()

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')









