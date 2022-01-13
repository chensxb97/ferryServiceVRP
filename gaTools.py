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
from utils import Color, Edges, Locations, computeDistMatrix

MapGraph = nx.Graph()
MapGraph.add_weighted_edges_from(Edges)

def cxPartiallyMatched(ind1, ind2):
    size = min(len(ind1), len(ind2))
    cxPoint1, cxPoint2 = sorted(random.sample(range(size), 2))
    temp1 = ind1[cxPoint1:cxPoint2+1] + ind2
    temp2 = ind1[cxPoint1:cxPoint2+1] + ind1
    ind1 = []
    for gene in temp1:
        if gene not in ind1:
            ind1.append(gene)
    ind2 = []
    for gene in temp2:
        if gene not in ind2:
            ind2.append(gene)
    return ind1, ind2

def drawGaSolution(route, df, ax):
    for i in range(len(route)):
        subroute = route[i]
        subroute.append(0)
        subroute.insert(0,0)
        print(subroute)
        for j in range(len(subroute)-1):
            ax.scatter(Locations[df.iloc[subroute[j], 2]][0], Locations[df.iloc[subroute[j], 2]][1], marker='o')
            zone_s = df.iloc[subroute[j], 2]
            print(zone_s)
            zone_e = df.iloc[subroute[j+1], 2]
            print(zone_e)
            launch_id = str(i+1)
            ax.arrow(Locations[zone_s][0], Locations[zone_s][1], Locations[zone_e][0]-Locations[zone_s][0], Locations[zone_e][1]-Locations[zone_s][1], head_width=10, head_length=10, color = Color[launch_id])

def evalVRP(individual, df, unit_cost=1.0, init_cost=0, wait_cost=0, delay_cost=0):
    total_cost = 0
    distMatrix =computeDistMatrix(df, MapGraph)
    route = ind2Route(individual, df)
    total_cost = 0
    Capacity = 14
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
                subRoute_distance =1000000
                subRoute_cost = 1000000
            # Update last customer ID
            lastCustomer_id = customer_id
        # Calculate transport cost
        subRoute_distance = subRoute_distance + distMatrix[lastCustomer_id][0]
        #sub_route_transport_cost = init_cost + unit_cost * sub_route_distance
        # Obtain sub-route cost
        subRoute_cost = subRoute_distance #sub_route_time_cost +
        subRoute_time_cost = subRoute_cost/0.463+service_time
        # Update total cost
        total_cost = total_cost + subRoute_distance
        if subRoute_time_cost > 150:
            total_cost += 100000000
    if len(route) <= 5:
        fitness = 1.0 / total_cost
    else:
        fitness = 0.000000001
    return (fitness, )

def ind2Route(individual, df):
    #print(individual)
    route = []
    # Initialize a sub-route
    subroute = []
    for customer_id in individual:
        if customer_id < df.shape[0] :
            # Add to current sub-route
            subRoute.append(customer_id)
        else:
            if subRoute != []:
            # Save current sub-route before return if not empty
                route.append(subRoute)
                subRoute=[]
    if subRoute !=[]:
        route.append(subRoute)
    return route

def mutInverseIndex(individual):
    start, stop = sorted(random.sample(range(len(individual)), 2))
    individual = individual[:start] + individual[stop:start-1:-1] + individual[stop+1:]
    return (individual, )

def printRoute(route, merge=False):
    route_str = '0'
    subRoute_count = 0
    for subRoute in route:
        subRoute_count += 1
        subRoute_str = '0'
        for customer_id in subRoute:
            subRoute_str = f'{subRoute_str} - {customer_id}'
            route_str = f'{route_str} - {customer_id}'
        subRoute_str = f'{subRoute_str} - 0'
        if not merge:
            print(f'  Vehicle {subRoute_count}\'s route: {subRoute_str}')
        route_str = f'{route_str} - 0'
    if merge:
        print(route_str)


