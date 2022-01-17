import sys
sys.path.append('C:/Program Files/IBM/ILOG/CPLEX_Studio201/cplex/python/3.7/x64_win64')
sys.path.insert(0,'C:/users/benedict/appdata/local/programs/python/python37/lib/site-packages')

import argparse
import datetime
import docplex.mp
from docplex.mp.model import Model
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import time as timer

from gaTools import ind2Route
from itertools import chain, permutations
from utils import Edges, computeDistMatrix, separateTasks

Capacity = 14
MapGraph = nx.Graph()
MapGraph.add_weighted_edges_from(Edges)

# Computes cost for the given permutation of zones
def evaluate(individual, df):

    # Initialise cost counter and inputs
    total_cost = 0
    earlyCost = 1
    lateCost = 1
    distMatrix =computeDistMatrix(df, MapGraph)
    route = ind2Route(individual, df)

    # Each subRoute is a route that is served by a launch
    for subRoute in route:
        subRoute_distance = 0
        subRoute_time = 540 # Start time of tour
        subRoute_penalty_cost = 0
        lastCustomer_id = 0
        initial_load = 0

        for i in range(len(subRoute)):
            if df.iloc[i, 1]==2:
                initial_load += df.iloc[i, 3] # Add delivery load to initial load
        subRoute_load = initial_load # Total delivery load
        
        # Customer_id: Zone
        for customer_id in subRoute:
            # Calculate travelling distance between zones
            distance = distMatrix[lastCustomer_id][customer_id]

            # Update sub-route distance and time
            subRoute_distance += distance
            subRoute_time += distance/0.463

            # Time windows
            ready_time = df.iloc[customer_id, 4]
            due_time = df.iloc[customer_id, 5]

            # Compute penalty costs upon arrival
            subRoute_penalty_cost += max(earlyCost*(ready_time-subRoute_time),0,lateCost*(subRoute_time-due_time))

            # Update load
            if df.iloc[customer_id, 1]==1:
                subRoute_load += df.iloc[customer_id, 3] # pickup
            else:
                subRoute_load -= df.iloc[customer_id, 3] # delivery

            # Update subRoute time after serving customer
            subRoute_time += df.iloc[customer_id, 3]

            # Capacity constraint
            if subRoute_load > Capacity: 
                subRoute_distance = 10000
            
            # Update last customer ID
            lastCustomer_id = customer_id

        # Total distance and time computed after returning to the depot
        returnToDepot = distMatrix[lastCustomer_id][0]
        subRoute_distance += returnToDepot
        subRoute_time += returnToDepot/0.463

        # Update total cost
        total_cost = total_cost + subRoute_distance + subRoute_penalty_cost

        # Tour duration balance constraint
        if subRoute_time > 690: # End time of tour
            total_cost += 10000

    return total_cost

def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--file', metavar='f', default='L1', help='file name of the order book that required to be processed')
    argparser.add_argument('--fleetsize', metavar='l', default='5', help='number of launches available')
    args = argparser.parse_args()
    dirName = os.path.dirname(os.path.abspath(__file__))
    file = args.file
    fileName = os.path.join(dirName, 'SampleDataset', file + '.csv')
    fleet = int(args.fleetsize)
    order_df = pd.read_csv(fileName, encoding='latin1', error_bad_lines=False)
    order_df = order_df.sort_values(by=['Start_TW','End_TW'])

    # Pre-optimistion step
    df_MSP, fleetsize_MSP, df_West, fleetsize_West = separateTasks(order_df, fleet)

    # Evalutate minimum cost for West
    list1 = [i for i in range(1, df_West.shape[0]+fleetsize_West)]
    perm1 = permutations(list1)
    cost1 = 10000
    for i in list(perm1):
        cost = evaluate(i,df_West)
        if cost < cost1:
            cost1 = cost
    print('Minimum cost for West tour:', cost1)

    # Evaluate minimum cost for MSP 
    list2 = [i for i in range(1, df_MSP.shape[0]+fleetsize_MSP)]
    perm2 = permutations(list2)
    cost2 = 10000
    for i in list(perm2):
        #print(i)
        cost = evaluate(i,df_MSP)
        if cost < cost2:
            cost2 = cost
    print('Minimum cost for MSP tour:', cost2)

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')




