import os
import io
import sys
sys.path.insert(0, '/usr/local/lib/python3.7/site-packages')
import random
from csv import DictWriter
from deap import base, creator, tools
import matplotlib.pyplot as plt
import networkx as nx
import datetime
import argparse
import pandas as pd
import numpy as np
import time
# the big 'M'
M = 10000
Capacity = 14

Color = {'1': 'b', '2': 'c', '3': 'k', '4': 'm', '5': 'r'}

Location = {"Z01" : [2726, 961], "Z02" : [2784, 1034], "Z03" : [2791, 1103], "Z04" : [2793, 1159], "Z05" : [2462, 1229], "Z06" : [2432, 1188], "Z07" : [2406, 1148], "Z08" : [2290, 1111], "Z09" : [2231, 1188], "Z10" : [2189, 1253], "Z11" : [2178, 1305], "Z12" : [2060, 1437], "Z13" : [1993, 1386], "Z14" : [1848, 1350], "Z15" : [1867, 1465], "Z16" : [1753, 1420], "Z17" : [1512, 1587], "Z18" : [1477, 1477], "Z19" : [1386, 1461], "Z20" : [1337, 1503], "Z21" : [1415, 1540], "Z22" : [1130, 1750], "Z23" : [1090, 1700], "Z24" : [1047, 1584], "Z25" : [956, 1578], "Z26" : [975, 1871], "Z27" : [940, 1830], "Z28" : [895, 1788], "Z29" : [837, 1743], "Z30" : [768, 1700], "Z31" : [704, 1689], "Z32" : [608, 1585], "Z33" : [573, 1531], "Z34" : [550, 1472], "Port West" : [1190, 1210], "Port MSP" : [1736, 1331]}

Edge_List = [('Port West', 'East Jurong', 3.6), ('East Jurong', 'West Jurong', 4.4), ('West Jurong', 'Z34', 9.5),
             ('Z34', 'Z33', 1), ('Z33', 'Z32', 1), ('West Jurong', 'Z33', 6), ('West Jurong', 'Z32', 7),
             ('Sinki', 'Z32', 7.3), ('East Jurong', 'Sinki', 8), ('West Kepple', 'Sinki', 10),
             ('Sinki', 'Z31', 7), ('Z31', 'Z30', 1.3), ('Z30', 'Z29', 1.3),
             ('Z29', 'Z28', 1.3), ('Z28', 'Z27', 1.3), ('Z27', 'Z26', 1.3), ('Z25', 'Sinki', 0.8), ('Z25', 'Z24', 2.7), ('Z24', 'Z23', 3),
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
             ('Z02', 'Z01', 2.3), ('Port MSP', 'Z14', 0.4), ('Sinki', 'Z29', 7.5), ('Z24', 'East Jurong', 10.2),
             ('Z27', 'Z22', 4),
             ('Z32', 'Z31', 4), ('Z24', 'Z22', 6), ('Z22', 'Jong', 10)]

Map_Graph = nx.Graph()
Map_Graph.add_weighted_edges_from(Edge_List)

def compute_dist_matrix(df, map):
    Num_of_cust = df.shape[0]
    dist_matrix = np.zeros((Num_of_cust+2, Num_of_cust+2))
    for i in range(Num_of_cust+2):
        for j in range(Num_of_cust+2):
            if i<Num_of_cust and j < Num_of_cust:
                dist_matrix[i][j] = nx.dijkstra_path_length(map, df['Zone'][i], df['Zone'][j])
            elif i<Num_of_cust and j>=Num_of_cust:
                dist_matrix[i][j] = nx.dijkstra_path_length(map, df['Zone'][i], df['Zone'][0])
            elif j<Num_of_cust and i >=Num_of_cust:
                dist_matrix[i][j] = nx.dijkstra_path_length(map, df['Zone'][0], df['Zone'][j])
#dist_matrix = dist_matrix.round()
    return dist_matrix

###########################################################
# -----------------------df format-----------------------
# | Order ID |     Request_Type    |    Zone   | Demand |
# | 0        | 0                   | Port Name | 0      |
# | N        | 1-pickup 2-delivery | Zone Name | Amount |
###########################################################
def ind2route(individual, df, dist_matrix):
    #print(individual)
    '''gavrptw.core.ind2route(individual, instance)'''
    route = []
    #depart_due_time = instance['depart']['due_time']
    # Initialize a sub-route
    sub_route = []
    for customer_id in individual:
        if customer_id < df.shape[0] :
            # Add to current sub-route
            sub_route.append(customer_id)
        else:
            if sub_route != []:
            # Save current sub-route before return if not empty
                route.append(sub_route)
                sub_route=[]
            # Save current sub-route
            #route.append(sub_route)
            # Initialize a new sub-route and add to it
            #print(sub_route)
    if sub_route !=[]:
        route.append(sub_route)
        # Update last customer ID
#last_customer_id = customer_id
        
#print(route)
#print('done')
    return route


def print_route(route, merge=False):
    '''gavrptw.core.print_route(route, merge=False)'''
    route_str = '0'
    sub_route_count = 0
    for sub_route in route:
        sub_route_count += 1
        sub_route_str = '0'
        for customer_id in sub_route:
            sub_route_str = f'{sub_route_str} - {customer_id}'
            route_str = f'{route_str} - {customer_id}'
        sub_route_str = f'{sub_route_str} - 0'
        if not merge:
            print(f'  Vehicle {sub_route_count}\'s route: {sub_route_str}')
        route_str = f'{route_str} - 0'
    if merge:
        print(route_str)


def eval_vrptw(individual, df, unit_cost=1.0, init_cost=0, wait_cost=0, delay_cost=0):
    '''gavrptw.core.eval_vrptw(individual, instance, unit_cost=1.0, init_cost=0, wait_cost=0,
        delay_cost=0)'''
    total_cost = 0
    dist_matrix =compute_dist_matrix(df, Map_Graph)
    route = ind2route(individual, df, dist_matrix)
    total_cost = 0
    Capacity = 14
    for sub_route in route:
        #sub_route_time_cost = 0
        sub_route_distance = 0
        elapsed_time = 0
        last_customer_id = 0
        initial_load = 0
        service_time = 0
        for i in range(len(sub_route)):
            if df.iloc[i, 1]==2:
                initial_load += df.iloc[i, 3]
        sub_route_load = initial_load
        for customer_id in sub_route:
            # Calculate section distance
            distance = dist_matrix[last_customer_id][customer_id]
            # Update sub-route distance
            sub_route_distance = sub_route_distance + distance
            if df.iloc[customer_id, 1]==1:
                sub_route_load += df.iloc[customer_id, 3]
            else:
                sub_route_load -= df.iloc[customer_id, 3]
            service_time+=df.iloc[customer_id, 3]
            if sub_route_load> Capacity:
                #fitness = 0
                sub_route_distance =1000000
                sub_route_cost = 1000000
            # Calculate time cost
            #arrival_time = elapsed_time + distance
            #time_cost = wait_cost * \
            #    max(instance[f'customer_{customer_id}']['ready_time'] - arrival_time, 0) + \
            #    delay_cost * \
            #    max(arrival_time - instance[f'customer_{customer_id}']['due_time'], 0)
            # Update sub-route time cost
            #sub_route_time_cost = sub_route_time_cost + time_cost
            # Update elapsed time
            
            # Update last customer ID
            last_customer_id = customer_id
        # Calculate transport cost
        sub_route_distance = sub_route_distance + dist_matrix[last_customer_id][0]
        #sub_route_transport_cost = init_cost + unit_cost * sub_route_distance
        # Obtain sub-route cost
        sub_route_cost = sub_route_distance #sub_route_time_cost +
        sub_route_time_cost = sub_route_cost/0.463+service_time
        # Update total cost
        total_cost = total_cost + sub_route_distance
        if sub_route_time_cost > 150:
            total_cost += 100000000
    if len(route) <= 5:
        fitness = 1.0 / total_cost
    else:
        fitness = 0.000000001

    return (fitness, )


def cx_partialy_matched(ind1, ind2):
    '''gavrptw.core.cx_partialy_matched(ind1, ind2)'''
    size = min(len(ind1), len(ind2))
    cxpoint1, cxpoint2 = sorted(random.sample(range(size), 2))
    temp1 = ind1[cxpoint1:cxpoint2+1] + ind2
    temp2 = ind1[cxpoint1:cxpoint2+1] + ind1
    ind1 = []
    for gene in temp1:
        if gene not in ind1:
            ind1.append(gene)
    ind2 = []
    for gene in temp2:
        if gene not in ind2:
            ind2.append(gene)
    return ind1, ind2


def mut_inverse_indexes(individual):
    '''gavrptw.core.mut_inverse_indexes(individual)'''
    start, stop = sorted(random.sample(range(len(individual)), 2))
    individual = individual[:start] + individual[stop:start-1:-1] + individual[stop+1:]
    return (individual, )


def seperate_tasks(df, Num_of_vehicle):
    df_MSP = df[df['Port'] == 'MSP']
    df_West = df[df['Port'] == 'West']
    #df_MSP = df_MSP.sort('Expected time', ascending=1)
    #df_West = df_West.sort('Expected time', ascending=1)
    len_MSP = len(df_MSP)
    len_West = len(df_West)
    fleetsize_MSP = round(len_MSP * Num_of_vehicle / (len_West+len_MSP))
    if fleetsize_MSP == Num_of_vehicle and len_West != 0:
        fleetsize_MSP -= 1
    fleetsize_West = Num_of_vehicle = fleetsize_MSP
    data = []
    data2 = []
    data.insert(0, {'Order_ID': 0, 'Request_Type': 0, 'Zone': 'Port MSP', 'Demand': 0, 'Port': 'MSP'})
    df_MSP = pd.concat([pd.DataFrame(data), df_MSP], ignore_index=True)
    data2.insert(0, {'Order_ID': 0, 'Request_Type': 0, 'Zone': 'Port West', 'Demand': 0, 'Port': 'West'})
    df_West = pd.concat([pd.DataFrame(data2), df_West], ignore_index=True)
    df_West = df_West.reset_index(drop=True)
    df_MSP = df_MSP.reset_index(drop=True)
    return df_MSP, fleetsize_MSP, df_West, fleetsize_West


'''wait_cost, delay_cost,'''
def run_gavrptw(df, unit_cost, init_cost,  ind_size, pop_size, \
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
    toolbox.register('evaluate', eval_vrptw, df=df, unit_cost=unit_cost, \
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
        #print(f'-- Generation {gen} --')
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
        #print(f'  Min {min(fits)}')
        #print(f'  Max {max(fits)}')
        #print(f'  Avg {mean}')
        #print(f'  Std {std}')
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
#print('-- End of (successful) evolution --')
    best_ind = tools.selBest(pop, 1)[0]
    print(f'Best individual: {best_ind}')
    print(f'Fitness: {best_ind.fitness.values[0]}')
    print_route(ind2route(best_ind, df, dist_matrix))
    print(f'Total cost: {1 / (best_ind.fitness.values[0])}')
    return best_ind


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--file', metavar='f', default='example', help='file name of the order book that required to be processed')
    #argparser.add_argument('--fleetsize', metavar='l', default='5', help='number of launches available')
    args = argparser.parse_args()
    filename = args.file + '.csv'
    Map_Graph = nx.Graph()
    Map_Graph.add_weighted_edges_from(Edge_List)
    df = pd.read_csv(filename)
    initial_time = time.time()
    df_MSP, fleetsize_MSP, df_West, fleetsize_West = seperate_tasks(df, 5)
    run_gavrptw(df_West, 1, 0, len(df_West)+1, 200,
                    0.85, 0.1, 80, export_csv=False, customize_data=False)
    
    mid_time = time.time()
    run_gavrptw(df_MSP, 1, 0, len(df_MSP)+1, 200,
            0.85, 0.1, 20, export_csv=False, customize_data=False)
    
    total_runtime = time.time()-initial_time
    print(mid_time-initial_time)
    print(time.time()-mid_time)
    print(total_runtime)
#eval_vrptw([]df_West)(individual, df, unit_cost=1.0, init_cost=0, wait_cost=0, delay_cost=0):

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
