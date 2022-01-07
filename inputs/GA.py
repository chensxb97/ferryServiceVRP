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
from ga_tools import ind2route, print_route, eval_vrp, cx_partialy_matched, mut_inverse_indexes, draw_ga_solution
from utils import compute_dist_matrix, separate_tasks, Location, Edge_List, Color

MUT_PROB = 0.1
CX_PROB = 0.85
GENERATION = 80
POPULATION_SIZE = 100
Capacity = 14

Map_Graph = nx.Graph()
Map_Graph.add_weighted_edges_from(Edge_List)

def run_gavrp(df, unit_cost, init_cost,  ind_size, pop_size, \
    cx_pb, mut_pb, n_gen, export_csv=False, customize_data=False):
    fitness_history = []
    gen_history = []

    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    # Attribute generator
    toolbox.register('indexes', random.sample, range(1, ind_size + 1), ind_size)
    # Structure initializers
    toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.indexes)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    # Operator registering
    toolbox.register('evaluate', eval_vrp, df=df, unit_cost=unit_cost, \
                     init_cost=init_cost)#, wait_cost=wait_cost, delay_cost=delay_cost
    toolbox.register('select', tools.selRoulette)
    toolbox.register('mate', cx_partialy_matched)
    toolbox.register('mutate', mut_inverse_indexes)
    pop = toolbox.population(n=pop_size)
    # Results holders for exporting results to CSV file
    csv_data = []
    print('Start of evolution')
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    print(f'  Evaluated {len(pop)} individuals')
    dist_matrix =compute_dist_matrix(df, Map_Graph)
    # Begin the evolution
    for gen in range(n_gen):
        print(f'-- Generation {gen} --')
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
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        #print(f'  Evaluated {len(invalid_ind)} individuals')
        # The population is entirely replaced by the offspring
        pop[:] = offspring
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        print(f'  Min {min(fits)}')
        print(f'  Max {max(fits)}')
        print(f'  Avg {mean}')
        print(f'  Std {std}')
        # Write data to holders for exporting results to CSV file
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
        fitness_history.append(1/mean)
        gen_history.append(gen)
    #plt.scatter(gen_history, fitness_history)
    print('-- End of (successful) evolution --')
    best_ind = tools.selBest(pop, 1)[0]
    print(f'Best individual: {best_ind}')
    print(f'Fitness: {best_ind.fitness.values[0]}')
    print_route(ind2route(best_ind, df, dist_matrix))
    print(f'Total cost: {1 / (best_ind.fitness.values[0])}')
    return best_ind

def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--file', metavar='f', default='M1', help='file name of the order book that required to be processed')
    argparser.add_argument('--fleetsize', metavar='l', default='5', help='number of launches available')
    args = argparser.parse_args()
    img = plt.imread("Singapore-Anchorages-Chart.png")
    fig, ax = plt.subplots()
    ax.imshow(img)
    dir_name = os.path.dirname(os.path.realpath('__file__'))
    file = args.file
    file_name = os.path.join(dir_name, 'SampleDataset', file + '.csv')
    Map_Graph = nx.Graph()
    Map_Graph.add_weighted_edges_from(Edge_List)
    df = pd.read_csv(file_name)
    initial_time = time.time()
    df_MSP, fleetsize_MSP, df_West, fleetsize_West = separate_tasks(df, 5)
    dist_matrix_1 =compute_dist_matrix(df_West, Map_Graph)
    dist_matrix_2 =compute_dist_matrix(df_MSP, Map_Graph)
    best_ind1 = run_gavrp(df_West, 1, 0, len(df_West)+1, POPULATION_SIZE,
                    CX_PROB, MUT_PROB, GENERATION, export_csv=False, customize_data=False)
    route1 = ind2route(best_ind1, df_West, dist_matrix_1)
    draw_ga_solution(route1, df_West, ax)
    mid_time = time.time()
    best_ind2 = run_gavrp(df_MSP, 1, 0, len(df_MSP)+1, POPULATION_SIZE,
                    CX_PROB, MUT_PROB, GENERATION, export_csv=False, customize_data=False)
    route2 = ind2route(best_ind2, df_MSP, dist_matrix_2)
    draw_ga_solution(route2, df_MSP, ax)
    total_runtime = time.time()-initial_time
    print(mid_time-initial_time)
    print(time.time()-mid_time)
    print(total_runtime)
#eval_vrptw([]df_West)(individual, df, unit_cost=1.0, init_cost=0, wait_cost=0, delay_cost=0):
    plt.show()
if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
