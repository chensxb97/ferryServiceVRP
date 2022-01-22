import sys
sys.path.insert(0,'C:/users/benedict/appdata/local/programs/python/python37/lib/site-packages')

from utils import Color, Locations

def drawSolution(solutionSet, df, ax):
    print('Drawing solution... ')
    for i in range(len(solutionSet)):
        ax.scatter(Locations[df.iloc[solutionSet[i][0], 2]][0], Locations[df.iloc[solutionSet[i][0], 2]][1], marker='o')
        zone_s = df.iloc[solutionSet[i][0], 2]
        zone_e = df.iloc[solutionSet[i][1], 2]
        launch_id = str(solutionSet[i][2])
        ax.arrow(Locations[zone_s][0], Locations[zone_s][1], \
            Locations[zone_e][0]-Locations[zone_s][0], Locations[zone_e][1]-Locations[zone_s][1], \
                head_width=10, head_length=10, color = Color[launch_id])