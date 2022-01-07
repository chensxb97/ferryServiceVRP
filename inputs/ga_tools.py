import sys
sys.path.insert(0,'C:/users/benedict/appdata/local/programs/python/python37/lib/site-packages')

import argparse
import datetime
import io
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import random
import time

from deap import base, creator, tools
from utils import Edge_List, Location, compute_dist_matrix, Color

Map_Graph = nx.Graph()
Map_Graph.add_weighted_edges_from(Edge_List)

def ind2route(individual, df, dist_matrix):
    #print(individual)
    route = []
    # Initialize a sub-route
    sub_route = []
    for customer_id in individual:
        if customer_id < df.shape[0] :
            # Add to current sub-route
            sub_route.append(customer_id)
        else:
            if sub_route != []:
            # Save current sub-route before return if not empty
                route.append(sub_route)
                sub_route=[]
    if sub_route !=[]:
        route.append(sub_route)
    return route


def print_route(route, merge=False):
    route_str = '0'
    sub_route_count = 0
    for sub_route in route:
        sub_route_count += 1
        sub_route_str = '0'
        for customer_id in sub_route:
            sub_route_str = f'{sub_route_str} - {customer_id}'
            route_str = f'{route_str} - {customer_id}'
        sub_route_str = f'{sub_route_str} - 0'
        if not merge:
            print(f'  Vehicle {sub_route_count}\'s route: {sub_route_str}')
        route_str = f'{route_str} - 0'
    if merge:
        print(route_str)


def draw_ga_solution(route, df, ax):
    for i in range(len(route)):
        subroute = route[i]
        subroute.append(0)
        subroute.insert(0,0)
        print(subroute)
        for j in range(len(subroute)-1):
            ax.scatter(Location[df.iloc[subroute[j], 2]][0], Location[df.iloc[subroute[j], 2]][1], marker='o')
            zone_s = df.iloc[subroute[j], 2]
            print(zone_s)
            zone_e = df.iloc[subroute[j+1], 2]
            print(zone_e)
            launch_id = str(i+1)
            ax.arrow(Location[zone_s][0], Location[zone_s][1], Location[zone_e][0]-Location[zone_s][0], Location[zone_e][1]-Location[zone_s][1], head_width=10, head_length=10, color = Color[launch_id])


def eval_vrp(individual, df, unit_cost=1.0, init_cost=0, wait_cost=0, delay_cost=0):
    total_cost = 0
    dist_matrix =compute_dist_matrix(df, Map_Graph)
    route = ind2route(individual, df, dist_matrix)
    total_cost = 0
    Capacity = 14
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
                sub_route_distance =1000000
                sub_route_cost = 1000000
            # Update last customer ID
            last_customer_id = customer_id
        # Calculate transport cost
        sub_route_distance = sub_route_distance + dist_matrix[last_customer_id][0]
        #sub_route_transport_cost = init_cost + unit_cost * sub_route_distance
        # Obtain sub-route cost
        sub_route_cost = sub_route_distance #sub_route_time_cost +
        sub_route_time_cost = sub_route_cost/0.463+service_time
        # Update total cost
        total_cost = total_cost + sub_route_distance
        if sub_route_time_cost > 150:
            total_cost += 100000000
    if len(route) <= 5:
        fitness = 1.0 / total_cost
    else:
        fitness = 0.000000001
    return (fitness, )


def cx_partialy_matched(ind1, ind2):
    size = min(len(ind1), len(ind2))
    cxpoint1, cxpoint2 = sorted(random.sample(range(size), 2))
    temp1 = ind1[cxpoint1:cxpoint2+1] + ind2
    temp2 = ind1[cxpoint1:cxpoint2+1] + ind1
    ind1 = []
    for gene in temp1:
        if gene not in ind1:
            ind1.append(gene)
    ind2 = []
    for gene in temp2:
        if gene not in ind2:
            ind2.append(gene)
    return ind1, ind2


def mut_inverse_indexes(individual):
    start, stop = sorted(random.sample(range(len(individual)), 2))
    individual = individual[:start] + individual[stop:start-1:-1] + individual[stop+1:]
    return (individual, )
