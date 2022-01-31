import sys
sys.path.append('C:/Program Files/IBM/ILOG/CPLEX_Studio201/cplex/python/3.7/x64_win64')
sys.path.insert(0,'C:/users/benedict/appdata/local/programs/python/python37/lib/site-packages')

import argparse
import networkx as nx
import os
import pandas as pd

from gaTools import ind2Route
from itertools import permutations
from utils import Edges, computeDistMatrix, separateTasks

Capacity = 14
MapGraph = nx.Graph()
MapGraph.add_weighted_edges_from(Edges)

# Computes cost for the given permutation of zones
def evaluate(individual, df,fleetsize):

    # Initialise cost counter and inputs
    total_cost = 0
    wait_cost = 1
    delay_cost = 1
    distMatrix =computeDistMatrix(df, MapGraph)
    route = ind2Route(individual, df)
    tourStart = df.iloc[0, 4]
    tourEnd = df.iloc[0,5]
    
    # Each subRoute is a route that is served by a launch
    for subRoute in route:
        initial_load = 0
        possibleCases = []
        heuristic = True

        for i in range(len(subRoute)):
            if df.iloc[i, 1]==2:
                initial_load += df.iloc[i, 3] # Add delivery load to initial load
        
        # Consider different cases
        for i in range(2): 
            subRoute_time = tourStart # Start time of tour
            subRoute_distance = 0
            subRoute_penalty_cost = 0
            lastCustomer_id = 0
            subRoute_load = initial_load # Total delivery load

            for customer_id in subRoute: # Customer_id: Zone
                # Calculate travelling distance between zones
                distance = distMatrix[lastCustomer_id][customer_id]

                # Update sub-route distance and time
                subRoute_distance += distance
                subRoute_time += distance/0.463

                # Time windows
                load = serv_time = df.iloc[customer_id,3]
                ready_time = df.iloc[customer_id, 4]
                due_time = df.iloc[customer_id, 5]
                
                if heuristic:
                    # Compute penalty costs
                    if ready_time > subRoute_time: # Launch is able to arrive at the ready time
                        subRoute_time = ready_time
                    else:
                        subRoute_penalty_cost += \
                        max(load*wait_cost*(ready_time-subRoute_time),0,load*delay_cost*(subRoute_time-due_time))
                else:
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
                    subRoute_distance += 1000000 # 7th digit
                
                # Update last customer ID
                lastCustomer_id = customer_id

            # Total distance and time computed after returning to the depot
            returnToDepot = distMatrix[lastCustomer_id][0]
            subRoute_distance += returnToDepot
            subRoute_time += returnToDepot/0.463

            # Tour duration balance constraint
            if subRoute_time > tourEnd: # End time of tour
                subRoute_distance += 10000000 # 8th digit

            possibleCases.append(subRoute_distance+subRoute_penalty_cost)
            
            # Change to no heuristic
            if heuristic:
                heuristic = False
        
        # Update total cost with the minimum value between the two cases
        total_cost = total_cost + min(possibleCases)

    # Maximum number of active launches cannot be more than assigned fleetsize
    if len(route) > fleetsize:
        total_cost = 100000000 # 9th digit

    return total_cost, route

def printOptimalRoute(best_route):
    for launch, launchRoute in enumerate(best_route,1):
        routeStr = '0'
        for zone in launchRoute:
            routeStr+= ' - '
            routeStr+= str(zone)
        routeStr+= ' - 0'
        print('Launch',launch, ':', routeStr)
        
def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--file', metavar='f', default='LT1', help='File name of test case')
    argparser.add_argument('--batch', metavar='b', default=True, help='Run all test cases from directory')
    argparser.add_argument('--fleetsize', metavar='l', default='5', help='Total number of launches available')
    args = argparser.parse_args()
    dirName = os.path.dirname(os.path.abspath(__file__))
    testFile = args.file
    batch = args.batch
    if batch:
        files = ['ML3', 'MR1', 'MR2', 'MR3','HT1', 'HT2', 'HT3','HM1','HM2','HM3', 'HL1', 'HL2', 'HL3', 'HR1', 'HR2', 'HR3']
    else:
        files = [testFile]
    for file in files:
        fileName = os.path.join(dirName, 'datasets', file + '.csv')
        fleet = int(args.fleetsize)
        order_df = pd.read_csv(fileName, encoding='latin1', error_bad_lines=False)
        order_df = order_df.sort_values(by=['Start_TW','End_TW'])

        # Pre-optimistion step
        df_MSP, fleetsize_MSP, df_West, fleetsize_West = separateTasks(order_df, fleet)

        # Evaluate minimum cost for West
        print(file)
        print('Port West')
        list1 = [i for i in range(1, df_West.shape[0]+fleetsize_West)]
        perm1 = permutations(list1)
        cost1 = 100000
        best_route1 = []
        all_routes = list(perm1)
        for i in all_routes:
            cost, route = evaluate(i,df_West,fleetsize_West)
            if cost < cost1:
                cost1 = cost
                best_route1 = route
        print('Optimal routes:')
        printOptimalRoute(best_route1)
        print('Minimum cost:', cost1)

        # Evaluate minimum cost for MSP 
        print('\nPort MSP')
        list2 = [i for i in range(1, df_MSP.shape[0]+fleetsize_MSP)]
        perm2 = permutations(list2)
        cost2 = 100000
        best_route2 = []
        all_routes = list(perm2)
        for i in all_routes:
            cost, route = evaluate(i,df_MSP,fleetsize_MSP)
            if cost < cost2:
                cost2 = cost
                best_route2 = route
        print('Optimal routes: ')
        printOptimalRoute(best_route2)
        print('Minimum cost:', cost2)
    

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')




