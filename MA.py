"""

Programmer: Trinav Bhattacharyya
Date of Development: 13/10/2020
This code has been developed according to the procedures mentioned in the following research article:
Zervoudakis, K., Tsafarakis, S., A mayfly optimization algorithm, Computers &
Industrial Engineering (2020)

"""
import numpy as np
from config import *


def MA(num_features,max_iter, num_agents,trainX, testX, trainy, testy , pop_pos_init, pop_fit_init,best_pop_init,best_fit_init,best_acc_init,best_cols_init,roound):

#(num_agents, train_data, train_label, obj_function=compute_fitness, trans_function_shape='s',  prob_mut=0.2,  save_conv_graph=False):
    
    # Mayfly Algorithm
    ############################### Parameters ####################################
    #                                                                             #
    #   num_agents: number of mayflies                                            #
    #   max_iter: maximum number of generations                                   #
    #   train_data: training samples of data                                      #
    #   train_label: class labels for the training samples                        #                
    #   obj_function: the function to maximize while doing feature selection      #
    #   prob_mut: probability of mutation                                         #
    #   trans_function_shape: shape of the transfer function used                 #
    #   save_conv_graph: boolean value for saving convergence graph               #
    #                                                                             #
    ###############################################################################

    prob_mut=0.2
    convergence_curve=[]
    
    # control parameters
    a1 = 1
    a2 = 1.5
    d = 0.1
    fl = 0.1
    g = 0.8
    beta = 2
    delta = 0.9
    
    # initialize position and velocities of male and female mayflies' and Leader (the agent with the max fitness)
    male_pos =pop_pos_init # initialize(num_agents, num_features)
    female_pos = pop_pos_init #initialize(num_agents, num_features)
    male_vel = np.random.uniform(low = -1, high = 1, size = (num_agents, num_features))
    female_vel = np.random.uniform(low = -1, high = 1, size = (num_agents, num_features))
    male_fitness = np.zeros((num_agents))
    male_accuracy = np.zeros(num_agents)
    female_fitness = np.zeros((num_agents))
    Leader_agent = np.zeros((num_features))
    Leader_fitness = best_fit_init
    Leader_accuracy = best_acc_init
    Leader_featuers=best_cols_init
    male_personal_best = np.zeros((num_agents, num_features))
    male_offspring = np.zeros((num_agents, num_features))
    female_offspring = np.zeros((num_agents, num_features))
    vmax_male = np.zeros((num_features))
    vmax_female = np.zeros((num_features))
    
    # rank initial population
    male_pos, male_fitness, Leader_fitness ,Leader_accuracy,Leader_featuers = sort_agents(male_pos,Leader_fitness,Leader_accuracy,Leader_featuers, trainX, testX, trainy, testy)
    female_pos, female_fitness, Leader_fitness, Leader_accuracy,Leader_featuers = sort_agents(female_pos,Leader_fitness,Leader_accuracy,Leader_featuers, trainX, testX, trainy, testy)    
  
    # main loop
    for iter_no in range(max_iter):
        print('MA,Itration : '+str(roound) +'-'+str(iter_no)+'  Fitness: '+str(Leader_fitness)+'  Acc: '+str(Leader_accuracy)+
                      '  NumF: '+str(len(Leader_featuers))+'  Features: '+str(Leader_featuers))


        #updating velocity limits
        vmax_male, vmax_female = update_max_velocity(male_pos, female_pos)
        
        for agent in range(num_agents):
            
            #updating Leader fitness and personal best fitnesses
            if male_fitness[agent] < Leader_fitness:
                Leader_fitness = male_fitness[agent]
                Leader_agent = male_pos[agent]
            
            if male_fitness[agent] < Fit_KNN(male_personal_best[agent],trainX, testX, trainy, testy)[0]:
                male_personal_best[agent] = male_pos[agent]

            #update velocities of male and female mayflies
            male_vel[agent], female_vel[agent] = update_velocity(male_pos[agent], female_pos[agent], male_vel[agent], female_vel[agent], Leader_agent, male_personal_best[agent], a1, a2, d, fl, g, beta, agent,trainX, testX, trainy, testy)
            
            #check boundary condition of velocities of male and female mayflies
            male_vel[agent], female_vel[agent] = check_velocity_limits(male_vel[agent], female_vel[agent], vmax_male, vmax_female)
            
            #applying transfer functions to update positions of male and female mayflies
            #the updation is done based on their respective velocity values
            for j in range(num_features):
                trans_value = sigmoid(male_vel[agent][j])
                if trans_value > np.random.normal(0,1):
                    male_pos[agent][j]=1
                else:
                    male_pos[agent][j]=0

                trans_value = sigmoid(female_vel[agent][j])
                if trans_value > np.random.random():
                    female_pos[agent][j]=1
                else:
                    female_pos[agent][j]=0
        
        #sorting 
        male_pos, male_fitness,Leader_fitness,Leader_accuracy,Leader_featuers = sort_agents(male_pos,Leader_fitness,Leader_accuracy,Leader_featuers,trainX, testX, trainy, testy)
        female_pos, female_fitness,Leader_fitness,Leader_accuracy,Leader_featuers = sort_agents(female_pos,Leader_fitness,Leader_accuracy,Leader_featuers, trainX, testX, trainy, testy)
        
        for agent in range(num_agents):
            
            #generation of offsprings by crossover and mutation between male and female parent mayflies
            male_offspring[agent], female_offspring[agent] = cross_mut(male_pos[agent], female_pos[agent],prob_mut)
            
        #comparing parents and offsprings and replacing parents wherever necessary
        male_pos = compare_and_replace(male_pos, male_offspring, male_fitness,Leader_fitness,Leader_accuracy,Leader_featuers,trainX, testX, trainy, testy)
        female_pos = compare_and_replace(female_pos, female_offspring, female_fitness,Leader_fitness,Leader_accuracy,Leader_featuers,trainX, testX, trainy, testy)
        
        #updating fitness values
        male_pos, male_fitness,Leader_fitness,Leader_accuracy,Leader_featuers = sort_agents(male_pos,Leader_fitness,Leader_accuracy,Leader_featuers,trainX, testX, trainy, testy)
        female_pos, female_fitness,Leader_fitness,Leader_accuracy,Leader_featuers = sort_agents(female_pos,Leader_fitness,Leader_accuracy,Leader_featuers,trainX, testX, trainy, testy)
        
        #updating values of nuptial dance
        d = d * delta
        fl = fl * delta
        
        #update final information
        if(male_fitness[0] < Leader_fitness):
            Leader_agent = male_pos[0].copy()
            Leader_fitness = male_fitness[0].copy()

        convergence_curve.append(np.mean(Leader_fitness))
    
    return  Leader_agent, Leader_fitness, convergence_curve,Leader_accuracy,Leader_featuers

def initialize(num_agents, num_features):
    # define min and max number of features
    min_features = int(0.3 * num_features)
    max_features = int(0.6 * num_features)

    # initialize the agents with zeros
    agents = np.zeros((num_agents, num_features))

    # select random features for each agent
    for agent_no in range(num_agents):

        # find random indices
        cur_count = np.random.randint(min_features, max_features)
        temp_vec = np.random.rand(1, num_features)
        temp_idx = np.argsort(temp_vec)[0][0:cur_count]

        # select the features with the ranom indices
        agents[agent_no][temp_idx] = 1   

    return agents

def sort_agents(agents,best_f,best_acc,best_featuers,trainX, testX, trainy, testy):
   
    if 1==1:
        # if there is only one agent
        if len(agents.shape) == 1:
            num_agents = 1
            fitness,tempacc,tempcols  = Fit_KNN(agents,trainX, testX, trainy, testy)
            if fitness[id]<best_f:
                best_f=fitness[id].copy()
                best_acc=tempacc
                best_featuers=tempcols
            return agents, fitness,best_f,best_acc,best_featuers

        # for multiple agents
        else:
            num_agents = agents.shape[0]
            fitness = np.zeros(num_agents)
            for id, agent in enumerate(agents):
                fitness[id],tempacc,tempcols  = Fit_KNN(agent,trainX, testX, trainy, testy)
                if fitness[id]<best_f:
                    best_f=fitness[id].copy()
                    best_acc=tempacc
                    best_featuers=tempcols

    idx = np.argsort(-fitness)
    sorted_agents = agents[idx].copy()
    sorted_fitness = fitness[idx].copy()

    return sorted_agents, sorted_fitness,best_f,best_acc,best_featuers

def update_max_velocity(male, female):
    size, length = male.shape
    agent1 = []
    agent2 = []
    r = np.random.normal(0,1 , size=(length))
    for j in range(length):
        r[j] *= 2
        agent1.append((male[0][j]-male[size-1][j])*r[j])
        agent2.append((female[0][j]-female[size-1][j])*r[j])
    
    return (agent1, agent2)

def update_velocity(m_pos, f_pos, m_vel, f_vel, Leader_agent, pbest, a1, a2, d, fl, g, b, i,trainX, testX, trainy, testy):
    tot_features = m_pos.shape[0]
    agent1 = np.zeros((tot_features))
    agent2 = np.zeros((tot_features))
    tot_features = len(m_pos)
    if i==0:
        for j in range(tot_features):
            agent1[j] = m_vel[j]+d*np.random.uniform(-1,1)
    else:
        sum = 0    
        for j in range(tot_features):
            sum = sum+(m_pos[j]-Leader_agent[j])*(m_pos[j]-Leader_agent[j])
        rg = np.sqrt(sum)
        sum = 0
        for j in range(tot_features):
            sum = sum+(m_pos[j]-pbest[j])*(m_pos[j]-pbest[j])
        rp = np.sqrt(sum)
        for j in range(tot_features):
            agent1[j] = g*m_vel[j]+a1*np.exp(-b*rp*rp)*(pbest[j]-m_pos[j])+a2*np.exp(-b*rg*rg)*(Leader_agent[j]-m_pos[j])
    if Fit_KNN(m_pos,trainX, testX, trainy, testy)[0] <= Fit_KNN(f_pos,trainX, testX, trainy, testy)[0]:
        sum = 0
        for j in range(tot_features):
            sum = sum+(m_pos[j]-f_pos[j])*(m_pos[j]-f_pos[j])
        rmf = np.sqrt(sum)
        agent2[j] = g*f_vel[j]+a2*np.exp(-b*rmf*rmf)*(m_pos[j]-f_pos[j])
    else:
        for j in range(tot_features):
            agent2[j] = g*f_vel[j]+fl*np.random.uniform(-1,1)
            
    return (agent1, agent2)

def check_velocity_limits(m_vel, f_vel, vmax_m, vmax_f):
    tot_features = len(m_vel)
    for j in range(tot_features):
        m_vel[j] = np.minimum(m_vel[j], vmax_m[j])
        m_vel[j] = np.maximum(m_vel[j], -vmax_m[j])
        f_vel[j] = np.minimum(f_vel[j], vmax_f[j])
        f_vel[j] = np.maximum(f_vel[j], -vmax_f[j])
    
    return (m_vel, f_vel)

def cross_mut(m_pos, f_pos,prob_mut):
    tot_features = len(m_pos)
    offspring1 = np.zeros((tot_features))
    offspring2 = np.zeros((tot_features))
    # partition defines the midpoint of the crossover
    partition = np.random.randint(tot_features//4, np.floor((3*tot_features//4)+1))

    # starting crossover
    for i in range(partition):
        offspring1[i] = m_pos[i]
        offspring2[i] = f_pos[i]

    for i in  range(partition, tot_features):
        offspring1[i] = f_pos[i]
        offspring2[i] = m_pos[i]
    # crossover ended


    # starting mutation
    if np.random.random() <= prob_mut:
        percent = 0.2
        numChange = int(tot_features*percent)
        pos = np.random.randint(0,tot_features-1,numChange)
        
        for j in pos:
            offspring1[j] = 1-offspring1[j]
        pos=np.random.randint(0,tot_features-1,numChange)
        for j in pos:
            offspring2[j] = 1-offspring2[j]

    # mutation ended
    
    if np.random.random() >= 0.5:
        return (offspring1, offspring2)
    else:
        return (offspring2, offspring1)


def compare_and_replace(pos, off, fit,Leader_fitness,Leader_accuracy,Leader_featuers,trainX, testX, trainy, testy):
    agents, features = pos.shape
    newfit = np.zeros((agents))
    temp_pos = np.zeros((agents, features))
    pos, fit,best_f,Leader_accuracy,Leader_featuers = sort_agents(pos,Leader_fitness,Leader_accuracy,Leader_featuers,trainX, testX, trainy, testy)
    # finding fitnesses of offsprings
    off, newfit,best_f,Leader_accuracy,Leader_featuers = sort_agents(off,Leader_fitness,Leader_accuracy,Leader_featuers,trainX, testX, trainy, testy)
    i=0
    j=0
    cnt=0
    # merging offsprings and parents and finding the next generation of mayflies
    while(cnt < agents):
        if fit[i] < newfit[j]:
            temp_pos[cnt] = pos[i].copy()
            i+=1
        else:
            temp_pos[cnt] = off[i].copy()
            j+=1
        cnt+=1
    return temp_pos

def sigmoid(val):
    if val < 0:
        return 1 - 1/(1 + np.exp(val))
    else:
        return 1/(1 + np.exp(-val))
