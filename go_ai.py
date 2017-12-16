from holdem import Teacher
import argparse
import time
import random
from multiprocessing import Process, Queue
import numpy as np
import pickle
seats = 8

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('n_instances', type=int, default=10)
    parser.add_argument('n_games', type=int, default=1)
    #termination criteria
    parser.add_argument('n_gens', type=int, default = 1000)
    parser.add_argument('fitness_threshold', type=float, default = 0.0)

    parser.add_argument('--quiet', dest='quiet', action='store_true')
    args = parser.parse_args()

    n_instances = args.n_instances
    n_games = args.n_games
    n_gens = args.n_gens
    quiet = args.quiet
    fitness_threshold = args.fitness_threshold

    retain_prob = 0.3
    rand_select = 0.15
    mutate_prob = 0.25
    mutate_gene_prob = 0.05

    
    inputfile1 = open("fitness_0.7_gen_6_of_50.pickle", "rb")
    inputfile2 = open("fitness_0.7_gen_26_of_50.pickle", "rb")
    inputfile3 = open("fitness_0.6666666666666667_gen_0_of_50.pickle", "rb")
    prev_results1 = pickle.load(inputfile1)
    prev_results2 = pickle.load(inputfile2)
    prev_results3 = pickle.load(inputfile3)

    #create initial population and get their fitness
    print("creating and running initial pop.")
    teachers = []
    q = Queue()
    for result in prev_results1:
        if len(teachers) == n_instances:
            break
        teachers.append(Teacher(q, 0, seats, n_games, quiet, result[1], result[2]))
    for result in prev_results2:
        if len(teachers) == n_instances:
            break
        if result[0] <= 0.7:
            teachers.append(Teacher(q, 0, seats, n_games, quiet, result[1], result[2]))
    for result in prev_results3:
        if len(teachers) == n_instances:
            break
        if result[0] <= 0.7:
            teachers.append(Teacher(q, 0, seats, n_games, quiet, result[1], result[2]))
    
    remaining = n_instances - len(teachers)
    for i in range(remaining):
        teachers.append(Teacher(q, i, seats, n_games, quiet))
    
    for teacher in teachers:
        teacher.start()
    '''
    for i in range(n_instances):
        teachers.append(Teacher(q, i, seats, n_games, quiet))
        teachers[i].start()
    '''
    results = []
    for i in range(n_instances):
        results.append(q.get())

    for i in range(n_instances):
        teachers[i].join()

    print("finished initial pop.")

    for g in range(n_gens):
        #gradient decent 0 fitness is optimal
        print("starting gen: ", g)
        results = [result for result in sorted(results, key = lambda x: x[0])]
        print("starting fitness of instances in this population:")
        for result in results:
            print(result[0], " - ", result[3])
        retain_len = int(retain_prob * n_instances)
        parents = []
        for i in range(retain_len):
            parents.append(Teacher(q, 0, seats, n_games, quiet, results[i][1], results[i][2]))
        mutations = 0
        for result in results[retain_len:]:
            if rand_select > random.random():
                mutations += 1
                #parents.append(Teacher(q, 0, seats, n_games, quiet, result[1], result[2]))
                parents.append(Teacher(q, 0, seats, n_games, quiet))
        print("total mutations for this gen: ", mutations)
        parents_len = len(parents)
        to_fill = n_instances - parents_len
        children = []
        while len(children) < to_fill:
            c1 = random.randint(0, parents_len - 1)
            c2 = random.choice(list(range(0, c1)) + list(range(c1 + 1, parents_len)))

            c1_weights = parents[c1].get_weights()
            c1_biases = parents[c1].get_biases()

            c2_weights = parents[c2].get_weights()
            c2_biases = parents[c2].get_biases()

            c3_weights = []
            c3_biases = []
            for x, y in zip(c1_weights, c2_weights):
                new_gene = []
                for x1, y1 in zip(x, y):
                    new_gene2 = []
                    for x2, y2 in zip(x1, y1):
                        new_gene2.append(random.choice([x2, y2]))
                    new_gene.append(new_gene2)
                c3_weights.append(np.array(new_gene))
            '''
            for x, y in zip(c1_weights, c2_weights):
                c3_weights.append(random.choice([x,y]))
            '''
            for x, y in zip(c1_biases, c2_biases):
                new_gene = []
                for x1, y1 in zip(x, y):
                    new_gene2 = []
                    for x2, y2 in zip(x1, y1):
                        new_gene2.append(random.choice([x2, y2]))
                    new_gene.append(new_gene2)
                c3_biases.append(np.array(new_gene))

            children.append(Teacher(q, 0, seats, n_games, quiet, c3_weights, c3_biases))

        teachers = parents + children

        for i in range(n_instances):
            teachers[i].start()

        results = []
        below_threshold = []
        for i in range(n_instances):
            result = q.get()
            results.append(result)
            if result[0] <= fitness_threshold:
                below_threshold.append(result)

        for i in range(n_instances):
            teachers[i].join()

        print("finished this gen.")
        #save the last generation
        if len(below_threshold) > 0:
            below_threshold = [result for result in sorted(below_threshold, key = lambda x: x[0])]
            print("got a champion")
            print("saving data to file")
            outfile = open("fitness_" + str(below_threshold[0][0]) + "_gen_" + str(g) + "_of_" + str(n_gens) + ".pickle", "wb")
            pickle.dump(results, outfile)
            outfile.close()

        if g == (n_gens - 1):
            results = [result for result in sorted(results, key = lambda x: x[0])]
            print("final generation fitnesses:")
            for result in results:
                print(result[0])
            print("saving data to file")
            outfile = open("fitness_" + str(results[0][0]) + "_gen_" + str(n_gens) + ".pickle", "wb")
            pickle.dump(results, outfile)
            outfile.close()