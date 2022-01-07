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
from lp_tools import print_solution
from utils import compute_dist_matrix, separate_tasks, Location, Edge_List, Color
# from scipy.spatial import distance_matrix

# the big 'M'
M = 10000
Capacity = 14
Map_Graph = nx.Graph()
Map_Graph.add_weighted_edges_from(Edge_List)

time_start = timer.time()

###########################################################
#-------------------------df format------------------------
# | Order ID |     Request_Type    |    Zone   | Demand |
# | 0        | 0                   | Port Name | 0      |
# | N        | 1-pickup 2-delivery | Zone Name | Amount |
###########################################################
def calculate_route(Num_of_cust, Num_of_vehicle, df):
    print(df)
    velocity = 0.463 #knot
    
    # create enumarator of 1 - N
    C = [i for i in range(1, Num_of_cust + 1)]
    # create enumarator of 0 - N
    Cc = [0] + C
    # create enumarator of 1 - V
    Num_of_vehicle = [i for i in range(1, Num_of_vehicle + 1)]

    # get distance matrix
    dist_matrix = compute_dist_matrix(df, Map_Graph)
    
    mdl = Model('VRP')
    # create variables
    p = [0]
    d = [0]
    ser = [0]
    
    # pickup & delivery volume
    for i in range(1, Num_of_cust+1):
        ser.append(df.iloc[i, 3])
        if df.iloc[i,1] == 2:
            d.append(df.iloc[i,3])
            p.append(0)
        else:
            d.append(0)
            p.append(df.iloc[i,3])

    # Variable set
    Load = [(i, v) for i in Cc for v in Num_of_vehicle]
    Index = [i for i in Cc]
    X = [(i, j, v) for i in Cc for j in Cc for v in Num_of_vehicle]

    # Calculate distance and time
    cost = {(i, j): dist_matrix[i][j] for i in Cc for j in Cc}
    time = {(i, j): dist_matrix[i][j]/velocity for i in Cc for j in Cc}

    # Creating variables
    x = mdl.binary_var_dict(X, name='x')
    load = mdl.integer_var_dict(Load, name='load')
    index = mdl.integer_var_dict(Index, name='index')

    # Defining Constraints

    # All vehicles will start at the depot
    mdl.add_constraints(mdl.sum(x[0, j, v] for j in Cc) == 1 for v in Num_of_vehicle)

    # All vehicles will return to depot
    mdl.add_constraints(mdl.sum(x[i, 0, v] for i in Cc) == 1 for v in Num_of_vehicle)

    # All nodes will only be visited once by one vehicle
    mdl.add_constraints(mdl.sum(x[i, j, v] for i in Cc for v in Num_of_vehicle if j != i) == 1 for j in C)

    # Vehicle will not terminate route anywhere except the depot
    mdl.add_constraints((mdl.sum(x[i, b, v] for i in Cc if i != b) - mdl.sum(x[b, j, v] for j in Cc if b != j)) == 0 for b in C for v in Num_of_vehicle)

    mdl.add_constraint(index[0] == 0)
    mdl.add_constraints(1 <= index[i] for i in C)
    mdl.add_constraints(Num_of_cust + 1 >= index[i] for i in C)
    mdl.add_constraints(index[i]-index[j]+1<=(Num_of_cust)*(1-x[i, j, v]) for i in C for j in C for v in Num_of_vehicle if i != j)

    # Vehicle initial load is the total demand for delivery in the route
    mdl.add_constraints((load[0, v] == mdl.sum(x[i, j, v]*d[j] for j in C for i in Cc if i != j)) for v in Num_of_vehicle)

    mdl.add_constraints((load[j, v] >= load[i, v] - d[j] + p[j] - M * (1 - x[i, j, v])) for i in Cc for j in C for v in Num_of_vehicle if i != j)

    # Total load does not exceed vehicle capacity
    mdl.add_constraints(load[j, v] <= Capacity for j in Cc for v in Num_of_vehicle)

    mdl.add_constraints(mdl.sum(x[i, j, v]*time[i, j] + x[i, j, v]*ser[i] for i in Cc for j in C)<=120 for v in Num_of_vehicle)
    mdl.add_constraints(mdl.sum(x[i, j, v]*time[i, j] + x[i, j, v]*ser[i] for i in C for j in Cc)<=120 for v in Num_of_vehicle)

    mdl.add_constraints(mdl.sum(x[i, j, v] for i in Cc for j in C)<=5 for v in Num_of_vehicle)
# Objective Function
# Minimize the total loss of revenue + cost
    obj_function = mdl.sum(cost[i, j] * x[i, j, v] for i in Cc for j in Cc for v in Num_of_vehicle if i !=j)

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

    actual_vehicle_usage = 0
    if solution != None:
        route = [x[i, j, k] for i in Cc for j in Cc for k in Num_of_vehicle if x[i, j, k].solution_value == 1]
        set = [[i, j, k] for i in Cc for j in Cc for k in Num_of_vehicle if x[i, j, k].solution_value == 1]
        for k in Num_of_vehicle:
            if x[0, 0, k].solution_value == 0:
                actual_vehicle_usage+=1
        print(set)
        print(obj_function.solution_value)
        return route, set, actual_vehicle_usage, obj_function.solution_value
    else:
        print('no feasible solution')
        return None, None, None, None

def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--file', metavar='f', default='example', help='file name of the order book that required to be processed')
    argparser.add_argument('--fleetsize', metavar='l', default='5', help='number of launches available')
    argparser.add_argument('--time', metavar = 't', default='540', help='starting time of optimization, stated in minutes; default at 9AM (540)')
    args = argparser.parse_args()
    img = plt.imread("Singapore-Anchorages-Chart.png")
    fig, ax = plt.subplots()
    ax.imshow(img)
    dir_name = os.path.dirname(os.path.realpath('__file__'))
    file = args.file
    file_name = os.path.join(dir_name, 'SampleDataset', file + '.csv')
    fleet = int(args.fleetsize)

    initial_order_df = pd.read_csv(file_name, encoding='latin1', error_bad_lines=False)

    df_MSP, fleetsize_MSP, df_West, fleetsize_West = separate_tasks(initial_order_df, fleet)
    
    route1, solution_set_West, used_fleet_West, cost1 = calculate_route(len(df_West)-1, 3, df_West)
    print_solution(solution_set_West, df_West, ax, 3)
    route2, solution_set_MSP, used_fleet_MSP, cost2= calculate_route(len(df_MSP)-1, 3, df_MSP)
    print_solution(solution_set_MSP, df_MSP, ax, 3)

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










