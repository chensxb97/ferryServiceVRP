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
from utils import compute_dist_matrix, separate_tasks, Location, Edge_List, Color

Map_Graph = nx.Graph()
Map_Graph.add_weighted_edges_from(Edge_List)

# draw solution on Map
def print_solution(solution_set, df, ax, fleetsize):
    for i in range(len(solution_set)):
        ax.scatter(Location[df.iloc[solution_set[i][0], 2]][0], Location[df.iloc[solution_set[i][0], 2]][1], marker='o')
        zone_s = df.iloc[solution_set[i][0], 2]
        zone_e = df.iloc[solution_set[i][1], 2]
        launch_id = str(solution_set[i][2])
        ax.arrow(Location[zone_s][0], Location[zone_s][1], Location[zone_e][0]-Location[zone_s][0], Location[zone_e][1]-Location[zone_s][1], head_width=10, head_length=10, color = Color[launch_id])

