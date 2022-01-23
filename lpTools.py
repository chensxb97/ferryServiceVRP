import sys
sys.path.insert(0,'C:/users/benedict/appdata/local/programs/python/python37/lib/site-packages')

from utils import Color, Locations

# Visualise solution on anchorage map           
def drawSolution(solutionSet, df, ax):
    for i in range(len(solutionSet)):
        ax.scatter(Locations[df.iloc[solutionSet[i][0], 2]][0], Locations[df.iloc[solutionSet[i][0], 2]][1], marker='o')
        zone_s = df.iloc[solutionSet[i][0], 2]
        zone_e = df.iloc[solutionSet[i][1], 2]
        launch_id = str(solutionSet[i][2])
        ax.arrow(Locations[zone_s][0], Locations[zone_s][1], \
            Locations[zone_e][0]-Locations[zone_s][0], Locations[zone_e][1]-Locations[zone_s][1], \
                head_width=10, head_length=10, color = Color[launch_id])

# Print each launch's route from solutionSet
def printRoutes(solutionSet):
    graphs = {}
    solutionSet.sort(key=lambda x:x[2])
    for launch_edge in solutionSet:
        i , j, launch = launch_edge[0], launch_edge[1], launch_edge[2]
        if launch not in graphs:
            graphs[launch] = {}
            graphs[launch][i] = [j]
        else:
            if i not in graphs[launch]:
                graphs[launch][i] = [j]
            else:
                graphs[launch][i].append(j)
    for launch, graph in graphs.items():
        routeStr = '0'
        next = graph[0][0]
        while next != 0:
            routeStr+=' - '
            routeStr+= str(next)
            next = graph[next][0]
        routeStr += ' - 0'
        print('Launch', launch, ':', routeStr)