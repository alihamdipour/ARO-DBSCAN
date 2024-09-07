# genetic algorithm search of the one max optimization problem
from numpy.random import randint
from numpy.random import rand
from config import *

# objective function
def onemax(x):
    return -sum(x)

# tournament selection
def selection(pop, scores, k=3):
    # first random selection
    selection_ix = randint(len(pop))
    for ix in randint(0, len(pop), k-1):
        # check if better (e.g. perform a tournament)
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]

# crossover two parents to create two children
def crossover(p1, p2, r_cross):
    # children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()
    # check for recombination
    if rand() < r_cross:
        # select crossover point that is not on the end of the string
        pt = randint(1, len(p1)-2)
        # perform crossover
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return [c1, c2]

# mutation operator
def mutation(bitstring, r_mut):
    for i in range(len(bitstring)):
        # check for a mutation
        if rand() < r_mut:
            # flip the bit
            bitstring[i] = 1 - bitstring[i]

# genetic algorithm
def GA(n_bits,n_iter, n_pop,trainX, testX, trainy, testy , pop_pos_init, pop_fit_init,best_pop_init,best_fit_init,best_acc_init,best_cols_init,roound):


    # crossover rate
    r_cross = 0.9
    # mutation rate
    r_mut = 1.0 / float(n_bits)
    con_cave=[best_fit_init]
    # initial population of random bitstring
    pop = [randint(0, 2, n_bits).tolist() for _ in range(n_pop)]
    # keep track of best solution
    best=0
    best_eval = best_fit_init
    scores=pop_fit_init
    bestAcc=best_acc_init
    bestFeatures=best_cols_init
    # enumerate generations
    for gen in range(n_iter-1):
        # evaluate all candidates in the population
        for i in range(n_pop):
            scores[i],tempAcc,tempFeatres=Fit_KNN(pop[i],trainX, testX, trainy, testy)
            if scores[i] < best_eval:
                best, best_eval = pop[i], scores[i]
                bestAcc=tempAcc
                bestFeatures=tempFeatres

                #print(">%d, new best f(%s) = %.3f" % (gen,  pop[i], scores[i]))
        # select parents
        selected = [selection(pop, scores) for _ in range(n_pop)]
        # create the next generation
        children = list()
        for i in range(0, n_pop, 2):
            # get selected parents in pairs
            p1, p2 = selected[i], selected[i+1]
            # crossover and mutation
            for c in crossover(p1, p2, r_cross):
                # mutation
                mutation(c, r_mut)
                # store for next generation
                children.append(c)
        # replace population
        pop = children
        con_cave.append(best_eval)
        print('GA,Itration : '+str(roound) +'-'+str(gen)+'  Fitness: '+str(best_eval)+'  Acc: '+str(bestAcc)+
              '  NumF: '+str(len(bestFeatures))+'  Features: '+str(bestFeatures))
    return best, best_eval, con_cave,bestAcc,bestFeatures
