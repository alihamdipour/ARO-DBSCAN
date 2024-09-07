import numpy as np
import time
from math import e
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn import datasets
from config import *
import copy

def sigmoid(val):
    if val < 0:
        return 1 - 1/(1 + np.exp(val))
    else:
        return 1/(1 + np.exp(-val))

def alturism(good_arr,bad_arr,good_vel,bad_vel):
    for i in range(len(good_vel)):
        if good_vel[i]>0 and good_vel[i]<1.5:
            if np.random.random()<np.random.uniform(0.5,0.8):
                bad_arr[i]=good_arr[i]
                bad_vel[i]=good_vel[i]
                good_vel[i]=np.random.random()
                trans_value = sigmoid(good_vel[i])
                if (np.random.random() < trans_value): 
                    good_arr[i] = 1
                else:
                    good_arr[i] = 0
        else:
            if np.random.random()<0.5:
                bad_arr[i]=good_arr[i]
                bad_vel[i]=good_vel[i]
                good_vel[i]=np.random.random()
                trans_value = sigmoid(good_vel[i])
                if (np.random.random() < trans_value): 
                    good_arr[i] = 1
                else:
                    good_arr[i] = 0
    return good_arr,bad_arr,good_vel,bad_vel

def AAPSO(num_features,max_iter,num_agents,trainX, testX, trainy, testy, pop_pos_init, pop_fit_init,best_pop_init,best_fit_init,best_acc_init,best_cols_init,roound):

    his_best_fit=[best_fit_init]
    fitness = pop_fit_init
    prev_fitness = np.zeros(num_agents)
    accuracy = np.zeros(num_agents)
    Leader_agent = best_pop_init
    Leader_fitness = best_fit_init
    Leader_accuracy = best_acc_init
    best_col=best_cols_init
    particles=pop_pos_init
 
 
    # rank initial particles
    # particles, fitness = sort_agents(particles, obj, data)
    fitness,particles  = (list(t) for t in zip(*sorted(zip(fitness, particles) ,key=lambda x:x[0])))


    # initialize global and local best particles
    globalBestParticle = [0 for i in range(num_features)]
    globalBestFitness = best_fit_init
    localBestParticle = [ [ 0 for i in range(num_features) ] for j in range(num_agents) ] 
    localBestFitness = [float("inf") for i in range(num_agents) ]
    weight = 1.0 
    velocity = [ [ 0 for i in range(num_features) ] for j in range(num_agents) ]
    
    for iter_no in range(max_iter-1):
        print('Itration : '+str(iter_no)+'-'+str(roound) +'  Fitness: '+str(Leader_fitness)+'  Acc: '+str(Leader_accuracy)+
              '  NumF: '+str(len(best_col))+'  Features: '+str(best_col))
        
        acc_list=np.zeros(num_agents)
        cols_list=[0]*num_agents
        # update adaptive weight
        weight= 1-(e**-(1-iter_no/max_iter))
        prev_fitness=fitness
        # update the velocity
        for i in range(num_agents):
            for j in range(num_features):
                velocity[i][j] = (weight*velocity[i][j])
                r1, r2 = np.random.random(2)
                velocity[i][j] = velocity[i][j] + (r1 * (localBestParticle[i][j] - particles[i][j]))
                velocity[i][j] = velocity[i][j] + (r2 * (globalBestParticle[j] - particles[i][j]))
       
        # updating position of particles
        for i in range(num_agents):
            for j in range(num_features):
                trans_value = sigmoid(velocity[i][j])
                if (np.random.random() < trans_value): 
                    particles[i][j] = 1
                else:
                    particles[i][j] = 0

        #alturism
        for i in range(num_agents):
            fitness[i],acc_list[i],cols_list[i]=Fit_KNN(particles[i],trainX, testX, trainy, testy)
        delta_fit=np.subtract(fitness,prev_fitness)
        alturism_rank=np.argsort(delta_fit)
        
        for i in range(int(0.3*num_agents)):
            good_idx=int((np.where(alturism_rank==(int(0.4*num_agents)+i+1)))[0])
            bad_idx=int((np.where(alturism_rank==num_agents-(i+1)))[0])
            particles[good_idx],particles[bad_idx],velocity[good_idx],velocity[bad_idx]=alturism(particles[good_idx],particles[bad_idx],velocity[good_idx],velocity[bad_idx])

        # updating fitness of particles
        # particles, fitness = sort_agents(particles, obj, data)
        fitness,particles,acc_list,cols_list  = (list(t) for t in zip(*sorted(zip(fitness, particles,acc_list,cols_list) ,key=lambda x:x[0])))
        
        # updating the global best and local best particles
        for i in range(num_agents):
            if fitness[i]<localBestFitness[i]:
                localBestFitness[i]=fitness[i]
                localBestParticle[i]=particles[i][:]

            if fitness[i]<globalBestFitness:
                globalBestFitness=fitness[i]
                globalBestParticle=particles[i][:]
                Leader_accuracy=acc_list[i]
                best_col=cols_list[i]

        # update Leader (best agent)
        if globalBestFitness < Leader_fitness:
            Leader_agent =copy.copy(globalBestParticle)
            Leader_fitness = copy.copy(globalBestFitness)

        his_best_fit.append(Leader_fitness)


    return  1,his_best_fit[-1],his_best_fit,Leader_accuracy,best_col
 
 