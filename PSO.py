import numpy as np
from torch import randperm
from matplotlib.pyplot import *
from pylab import *

from config import *
#https://github.com/zaman13/Particle-Swarm-Optimization-PSO-using-Python/blob/master/Code/pso_v1_1.py

def PSO(D,max_iter, Np,trainX, testX, trainy, testy , pop_pos_init, pop_fit_init,best_pop_init,best_fit_init,best_acc_init,best_cols_init,roound):
    

    w = 0.5                   # Intertial weight. In some variations, it is set to vary with iteration number.
    c1 = 2.0                  # Weight of searching based on the optima found by a particle
    c2 = 2.0                  # Weight of searching based on the optima found by the swarm
    v_fct = 1                 # Velocity adjust factor. Set to 1 for standard PSO.
    cov_cave=[best_fit_init]

    best_fit=best_fit_init


    pbest_val = best_fit_init           # Personal best fintess value. One pbest value per particle.
    gbest_val = np.zeros(max_iter)      # Global best fintess value. One gbest value per iteration (stored).

    best_acc=best_acc_init
    best_SF=best_cols_init
    tempAcc=np.zeros(Np)    
    tempSF= np.empty(Np, dtype=object)


    pbest = best_fit_init            # pbest solution
    gbest = np.zeros(D)                 # gbest solution

    gbest_store = np.zeros((max_iter,D))   # storing gbest solution at each iteration

    pbest_val_avg_store = np.zeros(max_iter)
    fitness_avg_store = np.zeros(max_iter)

    x = pop_pos_init           # Initial position of the particles
    v = np.zeros((Np,D))                # Initial velocity of the particles




        

    # Initial evaluations (for iteration = 0)
    # Function call. Evaluates the fitness of the initial swarms    
    fit=pop_fit_init         # vector of size Np

    pbest_val = np.copy(fit)   # initial personal best = initial fitness values. Vector of size Np
    pbest = np.copy(x)         # initial pbest solution = initial position. Matrix of size D x Np

    # Calculating gbest_val and gbest. Note that gbest is the best solution within pbest                                                                                                                      
    ind = np.argmin(pbest_val)                # index where pbest_val is min. 
    gbest_val[0] = best_fit_init   # set initial gbest_val
    gbest = best_pop_init
    pbest_val_avg_store[0] = np.mean(pbest_val)
    fitness_avg_store[0] = np.mean(fit)

    # Loop over the generations
    for iter in range(1,max_iter):

        print('PSO,Itration : '+str(roound) +'-'+str(iter)+'  Fitness: '+str(best_fit)+'  Acc: '+str(best_acc)+
              '  NumF: '+str(len(best_SF))+'  Features: '+str(best_SF))
    
        r1 = np.random.rand(Np,D)           # random numbers [0,1], matrix D x Np
        r2 = np.random.rand(Np,D)           # random numbers [0,1], matrix D x Np   
        v_global = np.multiply(((x-gbest)),r2)*c2*(-1.0)    # velocity towards global optima
        v_local = np.multiply((pbest- x),r1)*c1           # velocity towards local optima (pbest)

        v = w*v + (v_local + v_global)      # velocity update
    
        x = x + v*v_fct                     # position update
        
        for i in range(Np):
            fit[i],tempAcc[i],tempSF[i] = Fit_KNN(x[i][:],trainX, testX, trainy, testy)           # vector of size Np
                  # fitness function call (once per iteration). Vector Np
        
        # pbest and pbest_val update
        ind = np.argwhere(fit < pbest_val)  # indices where current fitness value set is greater than pbset
        pbest_val[ind] = np.copy(fit[ind])  # update pbset_val at those particle indices where fit > pbest_val
        pbest[ind,:] = np.copy(x[ind,:])    # update pbest for those particle indices where fit > pbest_val
    
        
        # gbest and gbest_val update
        ind2 = np.argmin(pbest_val)                       # index where the fitness is min
        gbest_val[iter] = np.copy(pbest_val[ind2])        # store gbest value at each iteration
        gbest = np.copy(pbest[ind2,:])                    # global best solution, gbest
        best_acc=tempAcc[ind2]
        best_SF=tempSF[ind2]

        if gbest_val[iter]<best_fit:
            best_fit=gbest_val[iter]
            best_acc=tempAcc[ind2]
            best_SF=tempSF[ind2]
        
        gbest_store[iter,:] = np.copy(gbest)              # store gbest solution
        
        pbest_val_avg_store[iter] = np.mean(pbest_val)
        fitness_avg_store[iter] = np.mean(fit)
        cov_cave.append(best_fit)
    return gbest, best_fit, cov_cave,best_acc,best_SF
