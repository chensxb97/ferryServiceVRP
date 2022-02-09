import sys
sys.path.insert(0,'C:/users/benedict/appdata/local/programs/python/python37/lib/site-packages')

import argparse
import matplotlib.pyplot as plt
import networkx as nx
import os
import pandas as pd
import random
import time

from deap import base, creator, tools
from gaTools import cxPartiallyMatched, drawGaSolution, evalVRP, ind2Route, mutInverseIndex, printRoute
from utils import Edges, separateTasks

MUT_PROB = 0.1
CX_PROB = 0.85
GENERATION = 80
POPULATION_SIZE = 100

Capacity = 14

MapGraph = nx.Graph()
MapGraph.add_weighted_edges_from(Edges)

# GA Algorithm
# @profile # Track memory usage
def runGA(df, fleetsize, unit_cost, init_cost,  ind_size, pop_size, \
    cx_pb, mut_pb, n_gen, export_csv=False, customize_data=False):
    
    fitnessHist = []
    genHistory = []
    creator.create('FitnessMax', base.Fitness, weights=(1.0, -1.0,-1.0))
    creator.create('Individual', list, fitness=creator.FitnessMax)

    # Registering GA variables
    toolbox = base.Toolbox()
    # Attribute generator
    toolbox.register('indexes', random.sample, range(1, ind_size + 1), ind_size)
    # Structure initializers
    toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.indexes)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    # Operator registering
    toolbox.register('evaluate', evalVRP, df=df, fleetsize=fleetsize, unit_cost=unit_cost, \
                     init_cost=init_cost)
    toolbox.register('select', tools.selRoulette)
    toolbox.register('mate', cxPartiallyMatched)
    toolbox.register('mutate', mutInverseIndex)
    pop = toolbox.population(n=pop_size)

    # Storing results prior to exporting as csv files
    csv_data = []

    print('Start of evolution')

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, (fit,fuel,penalty) in zip(pop, fitnesses):
        ind.fitness.values = (fit, fuel, penalty)

    # print(f'  Evaluated {len(pop)} individuals')

    # Begin the evolution
    for gen in range(n_gen):
        # print(f'-- Generation {gen} --')

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))

        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_pb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        for mutant in offspring:
            if random.random() < mut_pb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, (fit, fuel, penalty) in zip(invalid_ind, fitnesses):
            ind.fitness.values = (fit, fuel, penalty)

        # The population is entirely replaced by the offspring
        pop[:] = offspring
        
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        # print(f'  Min {min(fits)}')
        # print(f'  Max {max(fits)}')
        # print(f'  Avg {mean}')
        # print(f'  Std {std}')

        # Write results to csv variables prior to exporting as csv files
        if export_csv:
            csv_row = {
                'generation': gen,
                'evaluated_individuals': len(invalid_ind),
                'min_fitness': min(fits),
                'max_fitness': max(fits),
                'avg_fitness': mean,
                'std_fitness': std,
            }
            csv_data.append(csv_row)
        fitnessHist.append(1/mean)
        genHistory.append(gen)

    # plt.scatter(genHistory, fitnessHist)
    print('-- End of (successful) evolution --')
    best_ind = tools.selBest(pop, 1)[0]
    summaryGA(best_ind,df)

    return best_ind

def summaryGA(best_ind,df):
    print(f'Best individual: {best_ind}')
    print(f'Fitness: {best_ind.fitness.values[0]}')
    printRoute(ind2Route(best_ind, df))
    print(f'Minimum costs (Total, Fuel, Penalty): {1 / (best_ind.fitness.values[0])}, {best_ind.fitness.values[1]}, {best_ind.fitness.values[2]}')

def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--file', metavar='f', default='C1', help='File name of test case')
    argparser.add_argument('--batch', metavar='b', default=True, help='Run all test cases from directory')
    argparser.add_argument('--fleetsize', metavar='l', default='5', help='Total number of launches available')
    args = argparser.parse_args()
    testFile = args.file
    batch = args.batch
    fleet = int(args.fleetsize)

    # Directories
    dirName = os.path.dirname(os.path.abspath('__file__'))
    datasetsDir = os.path.join(dirName, 'datasets')
    outputsDir = os.path.join(dirName, 'outputs')
    outputsPlotsDir = os.path.join(outputsDir, 'plots', 'GA')
    if not os.path.exists(outputsPlotsDir):
        os.mkdir(outputsPlotsDir)

    # Anchorage map
    img = plt.imread("Port_Of_Singapore_Anchorages_Chartlet.png")
    
    if batch:
        testFiles = ['C1.csv','C2.csv','C3.csv','C4.csv','C5.csv','C6.csv', 'C7.csv', \
            'C8.csv','C9.csv','C10.csv','C11.csv','C12.csv', 'C13.csv', 'C14.csv']
        files = testFiles # All possible test cases
    else:
        testFile+= '.csv'
        files = [testFile] # Single test case

    for file in files:
        fileName = os.path.join(datasetsDir, file)

        # Dataset
        order_df = pd.read_csv(fileName, encoding='latin1', error_bad_lines=False)
        order_df = order_df.sort_values(by=['Start_TW','End_TW'])

        # Visualise map
        fig, ax = plt.subplots()
        ax.imshow(img)

        # Start time
        initial_time = time.time()

        # Pre-optimisation step
        df_MSP, fleetsize_MSP, df_West, fleetsize_West = separateTasks(order_df, fleet)

        # Run GA for West Tour
        best_ind1 = runGA(df_West,fleetsize_West, 1, 0, len(df_West)+1, POPULATION_SIZE,
                        CX_PROB, MUT_PROB, GENERATION, export_csv=False, customize_data=False)
        mid_time = time.time()
        
        # Run GA for MSP Tour
        best_ind2 = runGA(df_MSP,fleetsize_MSP, 1, 0, len(df_MSP)+1, POPULATION_SIZE,
                        CX_PROB, MUT_PROB, GENERATION, export_csv=False, customize_data=False)
        final_time = time.time()

        # Summary of results
        print(file)
        print('Port West')
        summaryGA(best_ind1,df_West)
        print('Time taken to run GA: ', mid_time - initial_time)
        print('\nPort MSP')
        summaryGA(best_ind2,df_MSP)
        print('Time taken to run GA: ', final_time - mid_time)

        # Visualise solutions
        route1 = ind2Route(best_ind1, df_West)
        route2 = ind2Route(best_ind2, df_MSP)
        drawGaSolution(route1, df_West, ax)
        drawGaSolution(route2, df_MSP, ax)

        # End time
        total_runtime = final_time - initial_time
        print('Total runtime: ', total_runtime)
        
        # plt.show()
        outputPlot = os.path.join(outputsPlotsDir, file.rsplit('.', 1)[0] + '.png')
        fig.savefig(outputPlot)
        print('\n')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
