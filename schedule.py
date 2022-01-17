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
from gaTools import drawGaSolution, ind2Route
from GA import runGA
from lpModel import calculateRoute
from lpTools import drawSolution
from utils import Edges, computeDistMatrix, separateTasks

MapGraph = nx.Graph()
MapGraph.add_weighted_edges_from(Edges)

# Start timer
time_start = timer.time()

# Results csv file
f = open('outputs/result_schedule.csv', 'w+')
f.write('L1,,L2,,L3,,L4,,L5\n')

########################################################################################################
#---------------------------------df format-------------------------------------------------------------
# | Order ID         |     Request_Type    |    Zone   | Demand |    Start_TW   |    End_TW  |   Port    
# | YYYY-MM-DD-ID    | 1-pickup 2-delivery | Zone Name | Amount | TourStart <=  x <= TourEnd | Port Name
########################################################################################################

# OLD VERSION
# def calculateRoute(numOfCustomers, numOfVehicles, df): 
#     print(df)
#     velocity = 0.463 #knot
    
#     # create enumarator of 1 - N
#     C = [i for i in range(1, numOfCustomers + 1)]
#     # create enumarator of 0 - N
#     Cc = [0] + C
#     # create enumarator of 1 - V
#     numOfVehicles = [i for i in range(1, numOfVehicles + 1)]
    
#     # get distance matrix
#     distMatrix = computeDistMatrix(df, MapGraph)
    
#     mdl = Model('VRP')
#     # create variables
#     p = [0]
#     d = [0]
#     ser = [0]
    
#     # pickup & delivery volume
#     for i in range(1, numOfCustomers+1):
#         ser.append(df.iloc[i, 3])
#         if df.iloc[i,1] == 2:
#             d.append(df.iloc[i,3])
#             p.append(0)
#         else:
#             d.append(0)
#             p.append(df.iloc[i,3])

#     # Variable set
#     Load = [(i, v) for i in Cc for v in numOfVehicles]
#     Index = [i for i in Cc]
#     X = [(i, j, v) for i in Cc for j in Cc for v in numOfVehicles]

#     # Calculate distance and time
#     cost = {(i, j): distMatrix[i][j] for i in Cc for j in Cc}
#     time = {(i, j): distMatrix[i][j]/velocity for i in Cc for j in Cc}
    
#     # Creating variables
#     x = mdl.binary_var_dict(X, name='x')
#     load = mdl.integer_var_dict(Load, name='load')
#     index = mdl.integer_var_dict(Index, name='index')
    
#     # Defining Constraints
    
#     # All vehicles will start at the depot
#     mdl.add_constraints(mdl.sum(x[0, j, v] for j in Cc) == 1 for v in numOfVehicles)
    
#     # All vehicles will return to depot
#     mdl.add_constraints(mdl.sum(x[i, 0, v] for i in Cc) == 1 for v in numOfVehicles)
    
#     # All nodes will only be visited once by one vehicle
#     mdl.add_constraints(mdl.sum(x[i, j, v] for i in Cc for v in numOfVehicles if j != i) == 1 for j in C)
    
#     # Vehicle will not terminate route anywhere except the depot
#     mdl.add_constraints((mdl.sum(x[i, b, v] for i in Cc if i != b) - mdl.sum(x[b, j, v] for j in Cc if b != j)) == 0 for b in C for v in numOfVehicles)
    
#     mdl.add_constraint(index[0] == 0)
#     mdl.add_constraints(1 <= index[i] for i in C)
#     mdl.add_constraints(numOfCustomers + 1 >= index[i] for i in C)
#     mdl.add_constraints(index[i]-index[j]+1<=(numOfCustomers)*(1-x[i, j, v]) for i in C for j in C for v in numOfVehicles if i != j)
    
#     # Vehicle initial load is the total demand for delivery in the route
#     mdl.add_constraints((load[0, v] == mdl.sum(x[i, j, v]*d[j] for j in C for i in Cc if i != j)) for v in numOfVehicles)
    
#     mdl.add_constraints((load[j, v] >= load[i, v] - d[j] + p[j] - M * (1 - x[i, j, v])) for i in Cc for j in C for v in numOfVehicles if i != j)
    
#     # Total load does not exceed vehicle capacity
#     mdl.add_constraints(load[j, v] <= Capacity for j in Cc for v in numOfVehicles)
    
#     mdl.add_constraints(mdl.sum(x[i, j, v]*time[i, j] + x[i, j, v]*ser[i] for i in Cc for j in C)<=120 for v in numOfVehicles)
#     mdl.add_constraints(mdl.sum(x[i, j, v]*time[i, j] + x[i, j, v]*ser[i] for i in C for j in Cc)<=120 for v in numOfVehicles)
    
#     mdl.add_constraints(mdl.sum(x[i, j, v] for i in Cc for j in C)<=5 for v in numOfVehicles)
#     # Objective Function
#     # Minimize the total loss of revenue + cost
#     obj_function = mdl.sum(cost[i, j] * x[i, j, v] for i in Cc for j in Cc for v in numOfVehicles if i !=j)
    
#     # Set time limit
#     mdl.parameters.timelimit.set(60)
    
#     # Solve
#     mdl.minimize(obj_function)
#     time_solve = timer.time()
#     solution = mdl.solve(log_output=True)
#     time_end = timer.time()
#     # print(solution)
#     running_time = round(time_end - time_solve, 2)
#     elapsed_time = round(time_end - time_start, 2)
    
#     actualVehicle_usage = 0
#     if solution != None:
#         route = [x[i, j, k] for i in Cc for j in Cc for k in numOfVehicles if x[i, j, k].solution_value == 1]
#         set = [[i, j, k] for i in Cc for j in Cc for k in numOfVehicles if x[i, j, k].solution_value == 1]
#         for k in numOfVehicles:
#             if x[0, 0, k].solution_value == 0:
#                 actualVehicle_usage+=1
#         print(set)
#         print(obj_function.solution_value)
#         return route, set, actualVehicle_usage, obj_function.solution_value
#     else:
#         print('no feasible solution')
#         return None, None, None, None

# Convert minutes to time
def minutes2Time(minutes):
    return str(int(minutes//60))+':'+ str(int(minutes%60))

# Print table to file
def printTable(table):
    for i in range(5):
        line = ''
        try:
            for j in range(5):
                try:
                    line += str(table[j][i][0])
                    line += ','
                    line += minutes2Time(table[j][i][1])
                    line += ','
                except IndexError or KeyError or TypeError:
                    line += ',,'
            print(line, file = f)
            f.write(line)
        except IndexError or KeyError or TypeError:
            pass

# Organise the results in a timetable
def route2Timetable(df, fleetsize, solutionSet, start_time):
    distMatrix = computeDistMatrix(df, MapGraph)
    route_set=[]
    for i in range(fleetsize):
        temp_list = []
        for j in range(len(solutionSet)):
            if solutionSet[j][2] == i+1:
                temp_list.append(solutionSet[j])
        route_set.append(temp_list)
    print(route_set)
    links = []
    for i in range(len(route_set)):
        temp_link = []
        start = 0
        for j in range(len(route_set[i])):
            for k in range(len(route_set[i])):
                if route_set[i][k][0]== start:
                    end=route_set[i][k][1]
                    temp_link.append(end)
                    start = end
                    break
        links.append(temp_link)
    print(links)
    print(df)
    print(distMatrix)
    locations = []
    time = []
    timetable = []
    for i in range(len(links)):
        temp_location = []
        temp_time = []
        if df['Port'][0]=='West':
            temp_location.append('West Coast Pier')
        else:
            temp_location.append('Marina South Pier')
        temp_time.append(start_time)
        last_time = start_time
        for j in range(len(links[i])):
            if links[i][j] != 0:
                temp_location.append(df['Zone'][links[i][j]])
                travel_time = round(distMatrix[links[i][j]][links[i][j-1]]/0.463)
                temp_time.append(travel_time+df['Demand'][links[i][j]]+last_time)
                last_time += travel_time+df['Demand'][links[i][j]]
        locations.append(temp_location)
        time.append(temp_time)
    print(locations)
    print(time)
    temp = []
    for i in range(len(locations)):
        temp_table = []
        for j in range(len(locations[i])):
            temp_table.append([locations[i][j], time[i][j]])
        temp.append(temp_table)
    return temp
    
def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--file', metavar='f', default='order_copy', help='file name of the order book that required to be processed')
    argparser.add_argument('--fleetsize', metavar='l', default='5', help='number of launches available')
    argparser.add_argument('--time', metavar = 't', default='540', help='starting time of optimization, stated in minutes; default at 9AM (540)') #0900 = 60*9 = 540
    args = argparser.parse_args()
    fig, ax = plt.subplots()
    dirName = os.path.dirname(os.path.abspath(__file__))
    file = args.file
    fileName = os.path.join(dirName, 'SampleDataset', file + '.csv')
    fleet = int(args.fleetsize)
    order_df = pd.read_csv(fileName, encoding='latin1', error_bad_lines=False)
    order_df = order_df.sort_values(by=['Start_TW','End_TW'])

    port = []
    for i in range(len(order_df)):
        print(order_df.iloc[i,0][11:13])
        order_df.iloc[i,0]=int(order_df.iloc[i,0][11:13]) # Truncate orderId to last 2 digits
        if int(order_df.iloc[i,2][1:3])>16: # Index [1:3] refers to the zone number
            port.append('West')
        else:
            port.append('MSP') # Zones 1-16 belong to Marina South Pier, while Zones 17-30 belong to West Coast Pier 
    order_df['Port']=port

    # df1, df2 = [x for _, x in order_df.groupby(order_df['Time'] >= 690)] # Grouping the dataset by df1(time < 690) and df2(time >= 690)
    # print(df1)
    # print(df2)
    # df=[df1.drop(['Time'], axis=1), df2.drop(['Time'], axis=1)] # Removing the time column and merging the datasets, creating a merged dataset that is sorted by time
    # This example illustrates only two tours. from 540-690 and from 690-840.
    # print(df) 

    # Split the orders according to tours, also ignores orders with unfeasible time windows
    tours = [(540,690), (690,840), (840,990), (990,1140)] # 0900-1130, 1130-1400, 1400-1630, 1630-1900
    df_tours = []
    i=1
    for tour in tours:
        df = order_df[(order_df['Start_TW'] >= tour[0]) & (order_df['End_TW'] <= tour[1])]
        if not df.empty:
            df_tours.append((tour,df))
            
    # Column names: Order_ID, Request_Type, Zone, Demand, Start_TW, End_TW, Port

    # Methodology
    # Attempt to solve using LP model for both clusters
    # If no solution is found using LP, run GA
    # Print solution and tabulate results in a timetable

    imgPath = os.path.join(dirName, 'Port_Of_Singapore_Anchorages_Chartlet.png')
    img = plt.imread(imgPath)

    for i in range(len(df_tours)):
        fig, ax = plt.subplots()
        ax.imshow(img)

        # Pre-optimisation step
        df_MSP, fleetsize_MSP, df_West, fleetsize_West = separateTasks(df_tours[i], fleet)

        route1, solutionSet_West, used_fleet_West, cost1 = calculateRoute(len(df_West)-1, fleetsize_West, df_West) # Perform LP
        if route1 == None: # No solution found
            while solution1_GA != None: # Perform GA
                solution1_GA = runGA(df_West, 1, 0, len(df_West)+1, 20, 0.85, 0.1, 20, export_csv=False, customize_data=False)
                drawGaSolution(ind2Route(solution1_GA, df_West), df_West, ax) # Draw GA solution
        else:
            drawSolution(solutionSet_West, df_West, ax, fleetsize_West) # Draw solution
            table_West = route2Timetable(df_West, fleetsize_West, solutionSet_West, 540+150*i) # Generate Timetable

        route2, solutionSet_MSP, used_fleet_MSP, cost2= calculateRoute(len(df_MSP)-1, fleetsize_MSP, df_MSP)
        if route2 == None:
            while solution2_GA != None:
                solution2_GA = runGA(df_MSP, 1, 0, len(df_MSP)+1, 20, 0.85, 0.1, 20, export_csv=False, customize_data=False)
                drawGaSolution(ind2Route(solution2_GA, df_MSP), df_MSP, ax)
        else:
            drawSolution(solutionSet_MSP, df_MSP, ax, fleetsize_MSP)
            table_MSP = route2Timetable(df_MSP, fleetsize_MSP, solutionSet_MSP, 540+150*i)
        
        plt.show() # Show routes

        # Consolidate both West and MSP timetables
        for i in range(len(table_MSP)):
            table_West.append(table_MSP[i])
        printTable(table_West)
            
    # Save timetable to csv file
    f.close()

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')










