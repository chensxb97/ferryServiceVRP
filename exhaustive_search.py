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

def evaluate(individual, df):
    total_cost = 0
    distMatrix =computeDistMatrix(df, MapGraph)
    route = ind2Route(individual, df)
    total_cost = 0
    for subRoute in route:
        #sub_route_time_cost = 0
        subRoute_distance = 0
        elapsed_time = 0
        lastCustomer_id = 0
        initial_load = 0
        service_time = 0
        for i in range(len(subRoute)):
            if df.iloc[i, 1]==2:
                initial_load += df.iloc[i, 3]
        subRoute_load = initial_load
        for customer_id in subRoute:
            # Calculate section distance
            distance = distMatrix[lastCustomer_id][customer_id]
            # Update sub-route distance
            subRoute_distance = subRoute_distance + distance
            if df.iloc[customer_id, 1]==1:
                subRoute_load += df.iloc[customer_id, 3]
            else:
                subRoute_load -= df.iloc[customer_id, 3]
            service_time+=df.iloc[customer_id, 3]
            if subRoute_load> Capacity:
                #fitness = 0
                subRoute_distance =10000
                subRoute_cost = 10000
            # Update last customer ID
            lastCustomer_id = customer_id
        # Calculate transport cost
        subRoute_distance = subRoute_distance + distMatrix[lastCustomer_id][0]
        # Obtain sub-route cost
        subRoute_cost = subRoute_distance #sub_route_time_cost +
        subRoute_time_cost = subRoute_cost/0.463+service_time
        # Update total cost
        total_cost = total_cost + subRoute_distance
        if subRoute_time_cost > 120:
            total_cost += 10000
    if len(route) > 5:
        total_cost = 10000
    return total_cost

def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--file', metavar='f', default='example', help='file name of the order book that required to be processed')
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
        #print(i)
        cost = evaluate(i,df_West) # CHANGE: evaluate
        if cost < cost1:
            cost1 = cost
    print('minimum cost for West:', cost1)

    # Evaluate minimum cost for MSP 
    list2 = [i for i in range(1, df_MSP.shape[0]+fleetsize_MSP)]
    perm2 = permutations(list2)
    cost2 = 10000
    for i in list(perm2):
        #print(i)
        cost = evaluate(i,df_MSP)
        if cost < cost2:
            cost1 = cost
    print('minimum cost for MSP:', cost2)

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')




