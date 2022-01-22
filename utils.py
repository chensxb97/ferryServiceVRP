import sys
sys.path.insert(0,'C:/users/benedict/appdata/local/programs/python/python37/lib/site-packages')

import networkx as nx
import numpy as np
import pandas as pd

Color = {'1': 'b', '2': 'c', '3': 'k', '4': 'm', '5': 'r'}

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

MapGraph = nx.Graph()
MapGraph.add_weighted_edges_from(Edges)

# Compute distance matrix
def computeDistMatrix(df, map):
    numOfCustomers = df.shape[0]
    distMatrix = np.zeros((numOfCustomers+2, numOfCustomers+2))
    for i in range(numOfCustomers+2):
        for j in range(numOfCustomers+2):
            if i<numOfCustomers and j < numOfCustomers:
                distMatrix[i][j] = nx.dijkstra_path_length(map, df['Zone'][i], df['Zone'][j]) # Interzonal distance
            elif i<numOfCustomers and j>=numOfCustomers:
                distMatrix[i][j] = nx.dijkstra_path_length(map, df['Zone'][i], df['Zone'][0]) # From zone to pier
            elif j<numOfCustomers and i >=numOfCustomers:
                distMatrix[i][j] = nx.dijkstra_path_length(map, df['Zone'][0], df['Zone'][j]) # From pier to zone
    return distMatrix

# Print all zone locations on map
def printMap(ax):
    for zone, pts in Locations.items():
        ax.scatter(pts[0],pts[1], marker='o')

# Launch assignment + Preparing inputs from bookings dataset
def separateTasks(order_df, numOfVehicles):
    if len(order_df) == 2:
        timeWindow = order_df[0]
        df = order_df[1]
    else:
        timeWindow = (540,690)
        df = order_df

    df = df.rename(columns={df.columns[0]:'Order_ID'})
    
    df_MSP = df[df['Port'] == 'MSP']
    df_West = df[df['Port'] == 'West']
    len_df_MSP = len(df_MSP)
    len_df_West = len(df_West)

    fleetsize_MSP = round(len_df_MSP * numOfVehicles / (len_df_West+len_df_MSP))
    if fleetsize_MSP == numOfVehicles and len_df_West != 0:
        fleetsize_MSP -= 1
    fleetsize_West = numOfVehicles - fleetsize_MSP
    
    data1 = pd.DataFrame([[0, 0, 'Port MSP', 0, timeWindow[0], timeWindow[1],'MSP']],columns=df_MSP.columns) # Departure Point (MSP)
    data2 = pd.DataFrame([[0, 0, 'Port West', 0, timeWindow[0], timeWindow[1], 'West']],columns=df_West.columns) # Departure Point (West)

    df_MSP = pd.concat([data1, df_MSP])
    df_MSP = df_MSP.reset_index(drop=True)
    df_West = pd.concat([data2, df_West])
    df_West = df_West.reset_index(drop=True)

    return df_MSP, fleetsize_MSP, df_West, fleetsize_West
    


