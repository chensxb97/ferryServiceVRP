import sys
sys.path.append('C:/Program Files/IBM/ILOG/CPLEX_Studio201/cplex/python/3.7/x64_win64')
sys.path.insert(0,'C:/users/benedict/appdata/local/programs/python/python37/lib/site-packages')

import argparse
import os
import pandas as pd

from gaTools import ind2Route
from itertools import permutations
from utils import MapGraph, computeDistMatrix, separateTasks

Capacity = 14

# Computes cost for the given permutation of zones
def evaluate(individual, df, fleetsize):

    # Initialise cost counter and inputs
    total_cost = 0
    total_distance = 0
    total_penalty_cost = 0
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
            subRoute_length = len(subRoute)

            for customer_position, customer_id in enumerate(subRoute): # Customer_id: Zone
                
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
                if heuristic:
                    if ready_time > subRoute_time: # Launch is able to arrive at the ready time
                        if customer_position < subRoute_length-1: # Current service location is not the last location
                            cur_penalty_cost = load*wait_cost*(ready_time-subRoute_time) # Penalty cost if the launch arrived early
                            next_load = df.iloc[subRoute[customer_position+1],3]
                            next_ready_time = df.iloc[subRoute[customer_position+1],4]
                            next_due_time = df.iloc[subRoute[customer_position+1],5]
                            next_penalty_cost = \
                                max(next_load*wait_cost*(ready_time+serv_time-next_ready_time),0,\
                                    next_load*delay_cost*(ready_time+serv_time-next_due_time)) # Penalty cost at the next service location if the launch arrived at the ready time
                            if cur_penalty_cost > next_penalty_cost: # Compares the penalty costs between the 2 service locations
                                subRoute_time = ready_time
                        else:
                            subRoute_time = ready_time
                
                subRoute_penalty_cost += max(load*wait_cost*(ready_time-subRoute_time),0,load*delay_cost*(subRoute_time-due_time))

                # Update load
                if df.iloc[customer_id, 1]==1:
                    subRoute_load += load # Pickup
                else:
                    subRoute_load -= load # Delivery

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

            # Maximum number of zones per launch constraint/Tour duration balance constraint
            if len(subRoute) > 5 or subRoute_time > tourEnd: # End time of tour
                subRoute_distance += 10000000 # 8th digit

            possibleCases.append((subRoute_distance+subRoute_penalty_cost,subRoute_distance,subRoute_penalty_cost))
            
            # Change to no heuristic
            if heuristic:
                heuristic = False
        
        # Update total cost with the minimum value between the two cases
        min_val = min(possibleCases, key = lambda t: t[0])
        total_cost += min_val[0]
        total_distance += min_val[1]
        total_penalty_cost += min_val[2]

    # Maximum number of active launches cannot be more than assigned fleetsize
    if len(route) > fleetsize:
        total_cost = 100000000 # 9th digit
        total_distance = 100000000
        total_penalty_cost = 0

    return total_cost, total_distance, total_penalty_cost, route

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
    argparser.add_argument('--file', metavar='f', default='LT1', help='File name of test case') # Change the filename to run a different test case
    argparser.add_argument('--batch', metavar='b', default=False, help='Run all test cases from directory') # Change to True to run batch of test cases
    argparser.add_argument('--fleetsize', metavar='l', default='5', help='Total number of launches available')
    args = argparser.parse_args()
    testFile = args.file
    batch = args.batch
    fleet = int(args.fleetsize)
    dirName = os.path.dirname(os.path.abspath(__file__))
    datasetsDir = os.path.join(dirName, 'datasets')

    if batch:
        testFiles = ['LT1.csv', 'LT2.csv', 'LL1.csv', 'LL2.csv', 'MT1.csv', 'MT2.csv', 'ML1.csv', 'ML2.csv',\
            'HT1.csv', 'HT2.csv', 'HL1.csv', 'HL2.csv', 'ET1.csv', 'ET2.csv', 'EL1.csv', 'EL2.csv',\
            'C1.csv','C2.csv','C3.csv','C4.csv','C5.csv','C6.csv', 'C7.csv','C8.csv',\
            'C9.csv','C10.csv','C11.csv','C12.csv', 'C13.csv', 'C14.csv'] # Full list of test cases
        files = testFiles # All possible test cases
    else:
        testFile+= '.csv'
        files = [testFile] # Single test case

    for file in files:
        fileName = os.path.join(datasetsDir, file)
        
        # Dataset
        order_df = pd.read_csv(fileName, encoding='latin1', error_bad_lines=False)
        order_df = order_df.sort_values(by=['Start_TW','End_TW'])

        # Pre-optimisation step
        df_MSP, fleetsize_MSP, df_West, fleetsize_West = separateTasks(order_df, fleet)

        # Evaluate minimum cost for West
        print(file)
        print('Port West')
        list1 = [i for i in range(1, df_West.shape[0]+fleetsize_West)]
        perm1 = permutations(list1)
        cost1 = 100000, 100000, 0
        best_route1 = []
        all_routes = list(perm1)
        for i in all_routes:
            cost, fCost, pCost, route = evaluate(i,df_West,fleetsize_West)
            if cost < cost1[0]:
                cost1 = cost, fCost, pCost
                best_route1 = route
        print('Optimal routes:')
        printOptimalRoute(best_route1)
        print(f'Minimum costs (Total, Fuel, Penalty): {cost1[0]}, {cost1[1]}, {cost1[2]}')

        # Evaluate minimum cost for MSP 
        print('\nPort MSP')
        list2 = [i for i in range(1, df_MSP.shape[0]+fleetsize_MSP)]
        perm2 = permutations(list2)
        cost2 = 100000, 100000, 0
        best_route2 = []
        all_routes = list(perm2)
        for i in all_routes:
            cost, fCost, pCost, route = evaluate(i,df_MSP,fleetsize_MSP)
            if cost < cost2[0]:
                cost2 = cost, fCost, pCost
                best_route2 = route
        print('Optimal routes: ')
        printOptimalRoute(best_route2)
        print(f'Minimum costs (Total, Fuel, Penalty): {cost2[0]}, {cost2[1]}, {cost2[2]}')

        print('\n')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')




