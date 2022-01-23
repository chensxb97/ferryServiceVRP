import sys
sys.path.insert(0,'C:/users/benedict/appdata/local/programs/python/python37/lib/site-packages')

import networkx as nx
import random

from utils import Color, Edges, Locations, computeDistMatrix

Capacity = 14
MapGraph = nx.Graph()
MapGraph.add_weighted_edges_from(Edges)

# PMX Technique
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

# Visualise solution on anchorage map
def drawGaSolution(route, df, ax):
    for i in range(len(route)):
        subroute = route[i]
        subroute.append(0)
        subroute.insert(0,0)
        for j in range(len(subroute)-1):
            ax.scatter(Locations[df.iloc[subroute[j], 2]][0], Locations[df.iloc[subroute[j], 2]][1], marker='o')
            zone_s = df.iloc[subroute[j], 2]
            zone_e = df.iloc[subroute[j+1], 2]
            launch_id = str(i+1)
            ax.arrow(Locations[zone_s][0], Locations[zone_s][1], \
                Locations[zone_e][0]-Locations[zone_s][0], Locations[zone_e][1]-Locations[zone_s][1], \
                    head_width=10, head_length=10, color = Color[launch_id])

# Evaluation algorithm
def evalVRP(individual, df, unit_cost=1.0, init_cost=0, wait_cost=1, delay_cost=1):

    # Initialise cost counter and inputs
    total_cost = 0
    distMatrix =computeDistMatrix(df, MapGraph)
    route = ind2Route(individual, df)
    tourStart = df.iloc[0, 4]
    tourEnd = df.iloc[0,5] 

    # Each subRoute is a route that is served by a launch
    for subRoute in route:
        subRoute_distance = 0
        subRoute_time = tourStart # Start time of tour
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
            load = serv_time = df.iloc[customer_id,3]
            ready_time = df.iloc[customer_id, 4]
            due_time = df.iloc[customer_id, 5]
            
            # Compute penalty costs
            subRoute_penalty_cost += max(load*wait_cost*(ready_time-subRoute_time),0,load*delay_cost*(subRoute_time-due_time))
            
            # Update load
            if df.iloc[customer_id, 1]==1:
                subRoute_load += load # pickup
            else:
                subRoute_load -= load # delivery

            # Update subRoute time after serving customer
            subRoute_time += serv_time

            # Capacity constraint
            if subRoute_load > Capacity: 
                subRoute_distance = 1000000
            
            # Update last customer ID
            lastCustomer_id = customer_id

        # Total distance and time computed after returning to the depot
        returnToDepot = distMatrix[lastCustomer_id][0]
        subRoute_distance += returnToDepot
        subRoute_time += returnToDepot/0.463

        # Update total cost
        total_cost = total_cost + subRoute_distance + subRoute_penalty_cost

        # Tour duration balance constraint
        if subRoute_time > tourEnd: # End time of tour
            total_cost += 100000000

    # Maximum number of zones in a tour
    if len(route) <= 5:
        fitness = 1.0 / total_cost
    else:
        fitness = 0.000000001

    return (fitness, )

# Generate route from individual
def ind2Route(individual, df):
    route = []
    subRoute = [] # Part of route
    for customer_id in individual:
        if customer_id < df.shape[0] : # Add zone to current subRoute
            subRoute.append(customer_id)
        else:
            if subRoute != []: # Save current subRoute before return if not empty
                route.append(subRoute)
                subRoute=[]
    if subRoute !=[]: # Save any remaining non-empty subRoute
        route.append(subRoute)
    return route

# Mutation function
def mutInverseIndex(individual):
    start, stop = sorted(random.sample(range(len(individual)), 2))
    individual = individual[:start] + individual[stop:start-1:-1] + individual[stop+1:]
    return (individual, )

# Print route
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
            print(f'Launch {subRoute_count}: {subRoute_str}')
        route_str = f'{route_str} - 0'
    if merge:
        print(route_str)


