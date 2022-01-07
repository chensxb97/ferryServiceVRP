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

from ga_tools import ind2route
from itertools import chain, permutations
from utils import compute_dist_matrix, separate_tasks, Edge_List
# from scipy.spatial import distance_matrix

Capacity = 14
Map_Graph = nx.Graph()
Map_Graph.add_weighted_edges_from(Edge_List)

def evaluate(individual, df):
    total_cost = 0
    dist_matrix =compute_dist_matrix(df, Map_Graph)
    route = ind2route(individual, df, dist_matrix)
    total_cost = 0
    for sub_route in route:
        #sub_route_time_cost = 0
        sub_route_distance = 0
        elapsed_time = 0
        last_customer_id = 0
        initial_load = 0
        service_time = 0
        for i in range(len(sub_route)):
            if df.iloc[i, 1]==2:
                initial_load += df.iloc[i, 3]
        sub_route_load = initial_load
        for customer_id in sub_route:
            # Calculate section distance
            distance = dist_matrix[last_customer_id][customer_id]
            # Update sub-route distance
            sub_route_distance = sub_route_distance + distance
            if df.iloc[customer_id, 1]==1:
                sub_route_load += df.iloc[customer_id, 3]
            else:
                sub_route_load -= df.iloc[customer_id, 3]
            service_time+=df.iloc[customer_id, 3]
            if sub_route_load> Capacity:
                #fitness = 0
                sub_route_distance =10000
                sub_route_cost = 10000
            # Update last customer ID
            last_customer_id = customer_id
        # Calculate transport cost
        sub_route_distance = sub_route_distance + dist_matrix[last_customer_id][0]
        # Obtain sub-route cost
        sub_route_cost = sub_route_distance #sub_route_time_cost +
        sub_route_time_cost = sub_route_cost/0.463+service_time
        # Update total cost
        total_cost = total_cost + sub_route_distance
        if sub_route_time_cost > 120:
            total_cost += 10000
    if len(route) > 5:
        total_cost = 10000
    return total_cost


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--file', metavar='f', default='example', help='file name of the order book that required to be processed')
    argparser.add_argument('--fleetsize', metavar='l', default='5', help='number of launches available')
    args = argparser.parse_args()
    dir_name = os.path.dirname(os.path.realpath('__file__'))
    file = args.file
    file_name = os.path.join(dir_name, 'SampleDataset', file + '.csv')
    fleet = int(args.fleetsize)
    
    initial_order_df = pd.read_csv(file_name, encoding='latin1', error_bad_lines=False)
    
    df_MSP, fleetsize_MSP, df_West, fleetsize_West = separate_tasks(initial_order_df, fleet)
    dist_matrix_1 =compute_dist_matrix(df_West, Map_Graph)
    dist_matrix_2 =compute_dist_matrix(df_MSP, Map_Graph)
    List1 = [i for i in range(1, df_West.shape[0]+fleetsize_West)]
    perm1 = permutations(List1)
    cost1 = 10000
    for i in list(perm1):
        #print(i)
        cost = evaluate(i,df_West)
        if cost < cost1:
            cost1 = cost
    print('minimum cost for West Coast Terminal Group:', cost1)
    List2 = [i for i in range(1, df_MSP.shape[0]+fleetsize_MSP)]
    perm2 = permutations(List2)
    cost2 = 10000
    for i in list(perm2):
        #print(i)
        cost = evaluate(i,df_MSP)
        if cost < cost2:
            cost1 = cost
    print('minimum cost for Marina South Pier Group:', cost2)

if __name__ == '__main__':
    
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')




