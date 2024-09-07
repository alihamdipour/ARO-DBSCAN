
import numpy as np
import pandas as pd
import math
import random

from config import *
 
 
def mutate(agent,tot_features):
            percent=0.2
            numChange=int(tot_features*percent)
            pos=np.random.randint(0,tot_features-1,numChange) #choose random positions to be mutated
            agent[pos]=1-agent[pos] #mutation
            return agent

def LAHC(particle, trainX, testX, trainy, testy,tot_features):
            delta=0.2
            _lambda = 3 #upper limit on number of iterations in LAHC
            target_fitness, target_acc, target_col = Fit_KNN(particle, trainX, testX, trainy, testy) #original fitness
            for i in range(_lambda):
                    new_particle = mutate(particle,tot_features) #first mutation
                    temp_fit,temp_acc,temp_col =Fit_KNN(new_particle, trainX, testX, trainy, testy)  
                    if temp_fit < target_fitness:
                        particle = new_particle.copy() #updation
                        target_fitness = temp_fit
                        target_acc=temp_acc
                        target_col=temp_col
                    elif (temp_fit<=(1+delta)*target_fitness):
                        temp_particle = new_particle.copy()
                        for j in range(_lambda):
                            temp_particle1 = mutate(temp_particle,tot_features) #second mutation
                            temp_fit2,temp_acc2,temp_col2 = Fit_KNN(temp_particle1, trainX, testX, trainy, testy)
                            if temp_fit2 < target_fitness:
                                target_fitness=temp_fit2
                                particle=temp_particle1.copy() #updation
                                target_acc=temp_acc2
                                target_col=temp_col2
                                break
            return particle,target_fitness, target_acc, target_col   
 

def transfer_func(velocity): #to convert into an array of zeros and ones
            t=[]
            for i in range(len(velocity)):
                    t.append(abs(velocity[i]/(math.sqrt(1+velocity[i]*velocity[i])))) #transfer function inside paranthesis
            return t
def SSD_LAHC(tot_features,max_iterations,swarm_size,trainX, testX, trainy, testy, pop_pos_init, pop_fit_init,best_pop_init,best_fit_init,best_acc_init,best_cols_init,roound):

#initialize swarm position and swarm velocity of SSD
    swarm_vel = np.random.uniform(low=0, high=1, size=(swarm_size,tot_features))
    his_best_fit=[best_fit_init]
    swarm_pos = pop_pos_init
    swarm_pos = np.where(swarm_pos>=0.5,1,0)

    c = 100
    alpha= 0.9

    gbest_fitness=best_fit_init
    pbest_fitness = np.zeros(swarm_size)
    pbest_fitness.fill(np.inf)  #initialize with the worse possible values
    pbest = np.empty((swarm_size,tot_features))
    gbest = np.empty(tot_features)
    pbest.fill(np.inf)
    gbest.fill(np.inf)

    bestCol=best_cols_init
    bestAcc=best_acc_init

    for itr in range(max_iterations-1):
                    
                    print('SSD, itration : '+str(roound)+'-'+str(itr)+'  Fitness: '+str(gbest_fitness)+'  Acc: '+str(bestAcc)+
                                        '  NumF: '+str(len(bestCol))+'  Features: '+str(bestCol))

                    for i in range(swarm_size):
                       
                        swarm_pos[i],fitness,acc,col = LAHC(swarm_pos[i],trainX, testX, trainy, testy,tot_features)   

                        if fitness < gbest_fitness:

                            gbest=swarm_pos[i].copy() #updating global best
                            gbest_fitness=fitness
                            bestAcc=acc
                            bestCol=col


                        if fitness < pbest_fitness[i]:
                            pbest[i] = swarm_pos[i].copy() #updating personal best
                            pbest_fitness[i]=fitness

                        r1 = random.random()
                        r2 = random.random()

                        #updating the swarm velocity
                        if r1 < 0.5:
                            swarm_vel[i] = c*math.sin(r2)*(pbest[i]-swarm_pos[i]) +math.sin(r2)* (gbest-swarm_pos[i])
                        else:
                            swarm_vel[i] = c*math.cos(r2)*(pbest[i]-swarm_pos[i]) + math.cos(r2)*(gbest-swarm_pos[i])
                        
                        #decaying value of c
                        c=alpha*c
                        
                        #applying transfer function and then updating the swarm position
                        t = transfer_func(swarm_vel[i])
                        for j in range(len(swarm_pos[i])):
                            if(t[j] < 0.5):
                                swarm_pos[i][j] = swarm_pos[i][j]
                            else:
                                swarm_pos[i][j] = 1 - swarm_pos[i][j]
                        
                    his_best_fit.append(gbest_fitness)



    selected_features = gbest
    print(gbest_fitness)
    return gbest, gbest_fitness, his_best_fit,bestAcc,bestCol
