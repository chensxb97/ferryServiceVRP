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

from csv import DictWriter
from deap import base, creator, tools

# # the big 'M'
M = 10000
Capacity = 14

Color = {'1': 'b', '2': 'c', '3': 'k', '4': 'm', '5': 'r'}

# Locations = {"Z01" : [2726, 961], "Z02" : [2784, 1034], "Z03" : [2791, 1103], "Z04" : [2793, 1159], "Z05" : [2462, 1229], "Z06" : [2432, 1188], "Z07" : [2406, 1148], "Z08" : [2290, 1111], "Z09" : [2231, 1188], "Z10" : [2189, 1253], "Z11" : [2178, 1305], "Z12" : [2060, 1437], "Z13" : [1993, 1386], "Z14" : [1848, 1350], "Z15" : [1867, 1465], "Z16" : [1753, 1420], "Z17" : [1512, 1587], "Z18" : [1477, 1477], "Z19" : [1386, 1461], "Z20" : [1337, 1503], "Z21" : [1415, 1540], "Z22" : [1130, 1750], "Z23" : [1090, 1700], "Z24" : [1047, 1584], "Z25" : [956, 1578], "Z26" : [975, 1871], "Z27" : [940, 1830], "Z28" : [895, 1788], "Z29" : [837, 1743], "Z30" : [768, 1700], "Z31" : [704, 1689], "Z32" : [608, 1585], "Z33" : [573, 1531], "Z34" : [550, 1472], "Port West" : [1190, 1210], "Port MSP" : [1736, 1331]}
# Edges = [('Port West', 'East Jurong', 3.6), ('East Jurong', 'West Jurong', 4.4), ('West Jurong', 'Z34', 9.5),
            #  ('Z34', 'Z33', 1), ('Z33', 'Z32', 1), ('West Jurong', 'Z33', 6), ('West Jurong', 'Z32', 7),
            #  ('Sinki', 'Z32', 7.3), ('East Jurong', 'Sinki', 8), ('West Kepple', 'Sinki', 10),
            #  ('Sinki', 'Z31', 7), ('Z31', 'Z30', 1.3), ('Z30', 'Z29', 1.3),
            #  ('Z29', 'Z28', 1.3), ('Z28', 'Z27', 1.3), ('Z27', 'Z26', 1.3), ('Z25', 'Sinki', 0.8), ('Z25', 'Z24', 2.7), ('Z24', 'Z23', 3),
            #  ('Z23', 'Z22', 1.8), ('West Kepple', 'Z20', 5), ('West Kepple', 'Z19', 10), ('West Kepple', 'Z18', 10),
            #  ('West Kepple', 'Jong', 3.3), ('Jong', 'Z20', 1.5), ('Jong', 'Z21', 3.5), ('Jong', 'Z17', 5.5),
            #  ('Jong', 'Southern', 9.3), ('Z20', 'Z19', 2), ('Z20', 'Z21', 2), ('Z19', 'Z18', 3), ('Z18', 'Z17', 3),
            #  ('Z17', 'Z21', 3), ('Z17', 'Sisters', 2), ('Sisters', 'Southern', 1.7), ('Southern', 'East Kepple', 5.2),
            #  ('Z16', 'East Kepple', 2.4), ('Z16', 'Z14', 2.4), ('Z14', 'Z15', 2.8), ('Z15', 'East Kepple', 1.6),
            #  ('Eastern Corridor', 'Z15', 2), ('Eastern Corridor', 'Z13', 2), ('Z13', 'Z12', 1.6),
            #  ('Z12', 'Eastern Corridor', 1.6), ('Z12', 'Eastern', 4), ('Z13', 'Eastern', 2), ('Z14', 'Eastern', 3),
            #  ('Z11', 'Eastern', 3), ('Z10', 'Eastern', 5), ('Z11', 'Z10', 2), ('Z10', 'Z09', 2.5),
            #  ('Z09', 'Z08', 1), ('Z10', 'Z05', 7), ('Z09', 'Z06', 6), ('Z08', 'Z07', 5), ('Z07', 'Z06', 1),
            #  ('Z06', 'Z05', 2.5), ('Z05', 'Z04', 8), ('Z06', 'Z04', 7.8), ('Z03', 'Z04', 2), ('Z03', 'Z02', 3.5),
            #  ('Z02', 'Z01', 2.3), ('Port MSP', 'Z14', 0.4), ('Sinki', 'Z29', 7.5), ('Z24', 'East Jurong', 10.2),
            #  ('Z27', 'Z22', 4),
            #  ('Z32', 'Z31', 4), ('Z24', 'Z22', 6), ('Z22', 'Jong', 10)]

'''
Remove 28, 30
Remove 33, 34
29 -> 28
31 -> 29
32 -> 30

West Jurong <- 34 <- 33 <- 32 <- 31 <- 30 <- 29 ...
30 <- 29 <- 28 <- 27
32 <- 31 <- (30<-29<-28) <- 27

'''
# New Locations and Edges dictionaries!!!
Locations = {"Z01" : [2710, 940], "Z02" : [2772, 1030], "Z03" : [2778, 1103], "Z04" : [2783, 1159], "Z05" : [2448, 1229], "Z06" : [2422, 1175], "Z07" : [2400, 1148], "Z08" : [2270, 1140], "Z09" : [2224, 1185], "Z10" : [2165, 1240], "Z11" : [2178, 1300], "Z12" : [2057, 1433], "Z13" : [1993, 1386], "Z14" : [1853, 1342], "Z15" : [1870, 1465], "Z16" : [1757, 1425], "Z17" : [1520, 1590], "Z18" : [1477, 1480], "Z19" : [1395, 1472], "Z20" : [1357, 1498], "Z21" : [1432, 1540], "Z22" : [1156, 1750], "Z23" : [1115, 1703], "Z24" : [1072, 1584], "Z25" : [973, 1575], "Z26" : [1005, 1871], "Z27" : [960, 1835], "Z28" : [867, 1747], "Z29" : [739, 1689], "Z30" : [643, 1585], "Port West" : [1211, 1200], "Port MSP" : [1740, 1325]}

Edges = [('Port West', 'East Jurong', 3.6), ('East Jurong', 'West Jurong', 4.4), ('West Jurong', 'Z30', 7),
             ('Sinki', 'Z30', 7.3), ('East Jurong', 'Sinki', 8), ('West Kepple', 'Sinki', 10),
             ('Sinki', 'Z29', 7), ('Z29', 'Z28', 2.6), ('Z28','Z27', 2.6), ('Z27', 'Z26', 1.3), 
             ('Z25', 'Sinki', 0.8), ('Z25', 'Z24', 2.7), ('Z24', 'Z23', 3),
             ('Z23', 'Z22', 1.8), ('West Kepple', 'Z20', 5), ('West Kepple', 'Z19', 10), ('West Kepple', 'Z18', 10),
             ('West Kepple', 'Jong', 3.3), ('Jong', 'Z20', 1.5), ('Jong', 'Z21', 3.5), ('Jong', 'Z17', 5.5),
             ('Jong', 'Southern', 9.3), ('Z20', 'Z19', 2), ('Z20', 'Z21', 2), ('Z19', 'Z18', 3), ('Z18', 'Z17', 3),
             ('Z17', 'Z21', 3), ('Z17', 'Sisters', 2), ('Sisters', 'Southern', 1.7), ('Southern', 'East Kepple', 5.2),
             ('Z16', 'East Kepple', 2.4), ('Z16', 'Z14', 2.4), ('Z14', 'Z15', 2.8), ('Z15', 'East Kepple', 1.6),
             ('Eastern Corridor', 'Z15', 2), ('Eastern Corridor', 'Z13', 2), ('Z13', 'Z12', 1.6),
             ('Z12', 'Eastern Corridor', 1.6), ('Z12', 'Eastern', 4), ('Z13', 'Eastern', 2), ('Z14', 'Eastern', 3),
             ('Z11', 'Eastern', 3), ('Z10', 'Eastern', 5), ('Z11', 'Z10', 2), ('Z10', 'Z09', 2.5),
             ('Z09', 'Z08', 1), ('Z10', 'Z05', 7), ('Z09', 'Z06', 6), ('Z08', 'Z07', 5), ('Z07', 'Z06', 1),
             ('Z06', 'Z05', 2.5), ('Z05', 'Z04', 8), ('Z06', 'Z04', 7.8), ('Z03', 'Z04', 2), ('Z03', 'Z02', 3.5),
             ('Z02', 'Z01', 2.3), ('Port MSP', 'Z14', 0.4), ('Sinki', 'Z28', 7.5), ('Z24', 'East Jurong', 10.2),
             ('Z27', 'Z22', 4), ('Z30', 'Z29', 4), ('Z24', 'Z22', 6), ('Z22', 'Jong', 10)]
# 
MapGraph = nx.Graph()
MapGraph.add_weighted_edges_from(Edges)

def computeDistMatrix(df, map):
    numOfCustomers = df.shape[0]
    distMatrix = np.zeros((numOfCustomers+2, numOfCustomers+2)) # distance matrix[point A][point B]
    for i in range(numOfCustomers+2):
        for j in range(numOfCustomers+2):
            if i<numOfCustomers and j < numOfCustomers:
                distMatrix[i][j] = nx.dijkstra_path_length(map, df['Zone'][i], df['Zone'][j]) # Interzonal distance
            elif i<numOfCustomers and j>=numOfCustomers:
                distMatrix[i][j] = nx.dijkstra_path_length(map, df['Zone'][i], df['Zone'][0]) # From zone to pier
            elif j<numOfCustomers and i >=numOfCustomers:
                distMatrix[i][j] = nx.dijkstra_path_length(map, df['Zone'][0], df['Zone'][j]) # From pier to zone
    return distMatrix
    
def separateTasks(order_df, numOfVehicles):
    if type(order_df) == 'list':
        tourStartEnd = order_df[0]
        df = order_df[1]
    else:
        tourStartEnd = (540,690)
        df = order_df
    df_MSP = df[df['Port'] == 'MSP']
    df_West = df[df['Port'] == 'West']
    len_df_MSP = len(df_MSP)
    len_df_West = len(df_West)

    fleetsize_MSP = round(len_df_MSP * numOfVehicles / (len_df_West+len_df_MSP))
    if fleetsize_MSP == numOfVehicles and len_df_West != 0:
        fleetsize_MSP -= 1
    fleetsize_West = numOfVehicles - fleetsize_MSP
    
    data1 = pd.DataFrame([[0, 0, 'Port MSP', 0, tourStartEnd[0], tourStartEnd[1],'MSP']],columns=df_MSP.columns)
    data2 = pd.DataFrame([[0, 0, 'Port West', 0, tourStartEnd[0], tourStartEnd[1], 'West']],columns=df_West.columns)
    # data1.insert(0, {'Order_ID': 0, 'Request_Type': 0, 'Zone': 'Port MSP', 'Demand': 0, 'Port': 'MSP', 'Start_Time': tourStartEnd[0],'End_Time': tourStartEnd[0]}) # Departure point
    # data2.insert(0, {'Order_ID': 0, 'Request_Type': 0, 'Zone': 'Port West', 'Demand': 0, 'Port': 'West','Start_Time': tourStartEnd[0],'End_Time': tourStartEnd[0]}) # Departure point
    df_MSP = pd.concat([data1, df_MSP])
    df_MSP = df_MSP.reset_index(drop=True)
    df_West = pd.concat([data2, df_West])
    df_West = df_West.reset_index(drop=True)

    return df_MSP, fleetsize_MSP, df_West, fleetsize_West
    
def printMap(ax):
    for zone, pts in Locations.items():
        ax.scatter(pts[0],pts[1], marker='o')