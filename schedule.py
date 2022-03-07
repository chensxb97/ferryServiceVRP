import sys
sys.path.append('C:/Program Files/IBM/ILOG/CPLEX_Studio201/cplex/python/3.7/x64_win64')
sys.path.insert(0,'C:/users/benedict/appdata/local/programs/python/python37/lib/site-packages')

import argparse
import matplotlib.pyplot as plt
import networkx as nx
import os
import pandas as pd
import time as timer

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
f = open('outputs/logs/schedule.csv', 'w+')
f.write('L1,,L2,,L3,,L4,,L5\n')

##############################################################################################
#---------------------------------df format---------------------------------------------------
# | Order_ID         |     Request_Type    |    Zone   | Demand |    Start_TW   |    End_TW  | 
# | YYYY-MM-DD-ID    | 1-pickup 2-delivery | Zone Name | Amount | TourStart <=  x <= TourEnd |
##############################################################################################

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

# Organise routes in a timetable
def route2Timetable(df, fleetsize, solutionSet):
    distMatrix = computeDistMatrix(df, MapGraph)
    route_set=[]
    start_time = df.iloc[0,4]
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
    argparser.add_argument('--file', metavar='f', default='order', help='File name of the order book')
    argparser.add_argument('--fleetsize', metavar='l', default='5', help='Total number of launches available')
    args = argparser.parse_args()
    file = args.file
    fleet = int(args.fleetsize)

    # Directory and File name
    dirName = os.path.dirname(os.path.abspath(__file__))
    fileName = os.path.join(dirName, 'datasets', file + '.csv')
    
    # Outputs directory
    outputsDir = os.path.join(dirName, 'outputs')
    outputsPlotsDir = os.path.join(outputsDir, 'plots','schedule')
    if not os.path.exists(outputsPlotsDir):
        os.mkdir(outputsPlotsDir)

    # Orders dataset
    order_df = pd.read_csv(fileName, encoding='latin1', error_bad_lines=False)
    order_df = order_df.sort_values(by=['Start_TW','End_TW'])

    # Anchorage map
    img = plt.imread("Port_Of_Singapore_Anchorages_Chartlet.png")
    
    # New 'Port' column
    port = []
    for i in range(len(order_df)):
        print(order_df.iloc[i,0][11:13])
        order_df.iloc[i,0]=int(order_df.iloc[i,0][11:13]) # Truncate orderId to last 2 digits
        if int(order_df.iloc[i,2][1:3])<=16: # Index [1:3] refers to the zone number
            port.append('MSP')
        else:
            port.append('West') # Zones 1-16 belong to Marina South Pier, while Zones 17-30 belong to West Coast Pier 
    order_df['Port']=port

    # Split the orders according to tours, also ignores orders with unfeasible time windows
    tours = [(540,690), (690,840), (840,990), (990,1140)] # 0900-1130, 1130-1400, 1400-1630, 1630-1900
    df_tours = []
    i=1
    for tour in tours:
        df = order_df[(order_df['Start_TW'] >= tour[0]) & (order_df['End_TW'] <= tour[1])]
        if not df.empty: # Store tour
            df_tours.append((tour,df)) 
    
    for i in range(len(df_tours)):
        fig, ax = plt.subplots()
        ax.imshow(img)

        # Pre-optimisation step
        df_MSP, fleetsize_MSP, df_West, fleetsize_West = separateTasks(df_tours[i], fleet)

        # Perform LP
        route1, solutionSet_West, _, _, _, _, _,_ = calculateRoute(len(df_West)-1, fleetsize_West, df_West) 

        # If no solution is found using LP, run GA
        # After solution is drawn, we generate a timetable for each route
        if route1 == None:
            while solution1_GA != None:
                solution1_GA = runGA(df_West, 1, 0, len(df_West)+1, 20, 0.85, 0.1, 20, export_csv=False, customize_data=False)
                drawGaSolution(ind2Route(solution1_GA, df_West), df_West, ax)
        else:
            drawSolution(solutionSet_West, df_West, ax)
            table_West = route2Timetable(df_West, fleetsize_West, solutionSet_West)

        route2, solutionSet_MSP, _, _, _, _, _, _= calculateRoute(len(df_MSP)-1, fleetsize_MSP, df_MSP)
        if route2 == None:
            while solution2_GA != None:
                solution2_GA = runGA(df_MSP, 1, 0, len(df_MSP)+1, 20, 0.85, 0.1, 20, export_csv=False, customize_data=False)
                drawGaSolution(ind2Route(solution2_GA, df_MSP), df_MSP, ax)
        else:
            drawSolution(solutionSet_MSP, df_MSP, ax)
            table_MSP = route2Timetable(df_MSP, fleetsize_MSP, solutionSet_MSP)
        
        plt.show() 
        outputPlot = os.path.join(outputsPlotsDir,file + '_' + str(i+1) + '_schedule.png')
        fig.savefig(outputPlot)

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










