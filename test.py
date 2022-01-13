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
from gaTools import cxPartiallyMatched, drawGaSolution, evalVRP, ind2Route, mutInverseIndex, printRoute
from lpTools import printSolution
from GA import runGA
# from scipy.spatial import distance_matrix
from utils import Color, Edges, Locations, computeDistMatrix, computeTravelTimeMatrix, separateTasks, printMap

import numpy as np
import cv2 #OpenCV library for python
from PIL import Image

def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--file', metavar='f', default='order', help='file name of the order book that required to be processed')
    argparser.add_argument('--fleetsize', metavar='l', default='5', help='number of launches available')
    argparser.add_argument('--time', metavar = 't', default='540', help='starting time of optimization, stated in minutes; default at 9AM (540)') #0900 = 60*9 = 540
    args = argparser.parse_args()
    fig, ax = plt.subplots()

    # dir_name = os.path.dirname(os.path.realpath('__file__')) 
    dirName = os.path.dirname(os.path.abspath(__file__)) # Path to directory
    file = args.file
    fileName = os.path.join(dirName, 'SampleDataset', file + '.csv')
#     fleet = int(args.fleetsize)
#     order_df = pd.read_csv(fileName, encoding='latin1', error_bad_lines=False)
#     port = []
#     for i in range(len(order_df)):
#         order_df.iloc[i,0]=int(order_df.iloc[i,0][11:13])
#         if int(order_df.iloc[i,2][1:3])>16:
#             port.append('West')
#         else:
#             port.append('MSP')
#     # CHANGE: Zones 1-16 belong to Marina South Pier, while Zones 17-30 belong to West Coast Pier 

#     order_df['Port']=port
#     df1, df2 = [x for _, x in order_df.groupby(order_df['Time'] >= 690)] # Grouping the dataset by df1(time < 690) and df2(time >= 690)
#     df=[df1.drop(['Time'], axis=1), df2.drop(['Time'], axis=1)] # Removing the time column and merging the datasets, creating a merged dataset that is sorted by time
#     # This example illustrates only two tours. from 540-690 and from 690-840.

#     # Methodology
#     # Solve by LP using calculate_route function for both clusters
#     # If no solution is found for both clusters, run GA
#     # Else, we print the solution found from LP
     
#     for i in range(len(df)): # For each tour
#         # imgPath = os.path.join(dirName, 'Singapore-Anchorages-Chart.png') # CHANGE: Update to line below
#         imgPath = os.path.join(dirName, 'Port_Of_Singapore_Anchorages_Chartlet.jpg')
#         img = plt.imread(imgPath)
#         fig, ax = plt.subplots()
#         ax.imshow(img)
#         df_MSP, fleetsize_MSP, df_West, fleetsize_West = separateTasks(df[i], fleet)
#         distMatrix1 =computeDistMatrix(df_West, MapGraph) # CHANGE: Update to lines below
#         distMatrix2 =computeDistMatrix(df_MSP, MapGraph) # CHANGE: Update to lines below
#         # travelTimeMatrix1 = computeTravelTimeMatrix(df_West,MapGraph)
#         # travelTimeMatrix2 = computeTravelTimeMatrix(df_MSP, MapGraph)
        
#         route1, solutionSet_West, used_fleet_West, cost1 = calculateRoute(len(df_West)-1, fleetsize_West, df_West) # CHANGE: LP Model
#         if route1 == None: # No solution found
#             while solution1_GA != None: # Perform GA
#                 solution1_GA = runGA(df_West, 1, 0, len(df_West)+1, 20, 0.85, 0.1, 20, export_csv=False, customize_data=False) # CHANGE: GA Algorithm 
#                 drawGaSolution(ind2Route(solution1_GA, df_West, distMatrix1), df_West, ax)
#         else: # Print LP Solution
#             printSolution(solutionSet_West, df_West, ax, fleetsize_West) # INVESTIGATE 
#             table_West = route2Timetable(df_West, fleetsize_West, solutionSet_West, 540+150*i, distMatrix1) # INVESTIGATE 

#         route2, solutionSet_MSP, used_fleet_MSP, cost2= calculateRoute(len(df_MSP)-1, fleetsize_MSP, df_MSP)
#         if route2 == None: # No solution found
#             while solution2_GA != None: # Perform GA
#                 solution2_GA = runGA(df_MSP, 1, 0, len(df_MSP)+1, 20, 0.85, 0.1, 20, export_csv=False, customize_data=False)
#                 drawGaSolution(ind2Route(solution2_GA, df_MSP, distMatrix2), df_MSP, ax)
#         else:
#             printSolution(solutionSet_MSP, df_MSP, ax, fleetsize_MSP)
#             table_MSP = route2Timetable(df_MSP, fleetsize_MSP, solutionSet_MSP, 540+150*i, distMatrix2)
#         plt.show()

# # consolidate timetable
#         for i in range(len(table_MSP)):
#             table_West.append(table_MSP[i])
#         printTable(table_West) # INVESTIGATE
            
# # save info to csv file as timetable
#     f.close()
    # imgPathBef = os.path.join(dirName, 'Singapore-Anchorages-Chart.png')
    imgPathAft = os.path.join(dirName, 'Port_Of_Singapore_Anchorages_Chartlet.png')
    # imgPathAft = os.path.join(dirName, 'original-4.jpg')
    img = plt.imread(imgPathAft)
    fig, ax = plt.subplots()
    ax.imshow(img)
    printMap(ax)
    plt.show()
    # im1 = cv2.imread(imgPathBef) # 2296, 3300
    # im2 = cv2.imread(imgPathAft) # 797, 1190

    # im = [im1,im2]
    # for i in im:
    #     print(type(i))
    #     # <class 'numpy.ndarray'>
    #     print(i.shape[0])
    #     print(i.shape[1])
    #     print(i.shape)
    #     print(type(i.shape))

    # image = Image.open(imgPathAft)
    # new_image = image.resize((3285, 2165))
    # new_image.save(os.path.join(dirName, 'Port_Of_Singapore_Anchorages_Chartlet.png'))

    # print(image.size) # Output: (1920, 1280)
    # print(new_image.size) # Output: (400, 400)

# Locations = {"Z01" : [2726, 961], "Z02" : [2784, 1034], "Z03" : [2791, 1103], "Z04" : [2793, 1159], "Z05" : [2462, 1229], "Z06" : [2432, 1188], 
# "Z07" : [2406, 1148], "Z08" : [2290, 1111], "Z09" : [2231, 1188], "Z10" : [2189, 1253], "Z11" : [2178, 1305], "Z12" : [2060, 1437], "Z13" : [1993, 1386], 
# "Z14" : [1848, 1350], "Z15" : [1867, 1465], "Z16" : [1753, 1420], "Z17" : [1512, 1587], "Z18" : [1477, 1477], "Z19" : [1386, 1461], "Z20" : [1337, 1503],
#  "Z21" : [1415, 1540], "Z22" : [1130, 1750], "Z23" : [1090, 1700], "Z24" : [1047, 1584], "Z25" : [956, 1578], "Z26" : [975, 1871], "Z27" : [940, 1830], 
#  "Z28" : [895, 1788], "Z29" : [837, 1743], "Z30" : [768, 1700], "Z31" : [704, 1689], "Z32" : [608, 1585], "Z33" : [573, 1531], "Z34" : [550, 1472], 
#  "Port West" : [1190, 1210], "Port MSP" : [1736, 1331]}
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


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')