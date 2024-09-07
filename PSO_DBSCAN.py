import numpy as np
from torch import randperm
from matplotlib.pyplot import *
from pylab import *
from sklearn.cluster import DBSCAN


from config import *

def params(pop_pos,npop,it): 
    alfa=npop-1
    kdis=[]
    for a in pop_pos:
        dis=[]
        for b in pop_pos:
            dis.append(math.dist(a,b))
        dis.sort()
        for k in range(1,alfa+1):
            kdis.append(dis[k])
    kdis.sort()
    # print(len(kdis))
    p1=[npop*(alfa-1)-1,kdis[npop*(alfa-1)-1]]
    p2=[npop*alfa-1,kdis[npop*alfa-1]]
    A1=(p2[1]-p1[1])/(p2[0]-p1[0])
    B1=p1[1]-A1*p1[0]
    A2=-1*A1
    B2=A1*(p1[0]+p2[0])+B1
    
    lowest_dif=1000000
    intersectionX=0
    for i in range(p1[0],p2[0]): 
        dif=abs((A2*i+B2)-kdis[i])
        if(dif<lowest_dif):
            lowest_dif=dif
            intersectionX=i
    p3=(intersectionX,kdis[intersectionX])
    M=[]
    A3=(kdis[p3[0]-2]-p3[1])/((p3[0]-2)-p3[0])
    B3=p3[1]-A3*p3[0]
    for i in range(p3[0],p2[0]):
        M.append(abs(kdis[i]-(A3*i+B3)))
    p_a=[0,0]
    p_a[1]=average(M)
    min_p_a=1000000
    for i in range(len(M)):
        dif=abs(p_a[1]-M[i])
        if(dif<min_p_a):
            min_p_a=dif
            p_a[0]=i

    d_p=(math.dist(p2,p3))/(math.dist(p1,p3))    
    b=int((p_a[0]+len(M))/2)
    if d_p>=4:
        eps=kdis[p3[0]+b]
    else:
        eps=kdis[p3[0]+p_a[0]]
    min_Pts=math.ceil(d_p-0.5)
    if min_Pts<1:
        min_Pts=1  
    return eps,min_Pts

def update_DBSCAN(popPos,popfit,it,trainX, testX, trainy, testy,best_x,best_f, best_acc,best_cols):

    popfit,popPos  = (list(t) for t in zip(*sorted(zip(popfit, popPos) ,key=lambda x:x[0])))
    radus,min_Pts=params(popPos,len(popPos),it)
    db = DBSCAN(eps=radus+0.00000001, min_samples=min_Pts).fit(popPos)
    labels = db.labels_

    for i in range(1,len(popPos)):         
        indexs = [m for m ,e in enumerate(labels) if e == labels[i]]
        if  i!=indexs[0]  and len(indexs)>2 and labels[i]!=-1 :
            tempPos1=deepcopy(popPos[i])   
            tempPos2=deepcopy(popPos[i])            
         
            tempPos1+=popPos[indexs[0]]-popPos[indexs[1]]/radus
            tempPos2-=popPos[indexs[0]]-popPos[indexs[1]]/radus
            tempFit1,tempAcc1,tempCols1=Fit_KNN(tempPos1,trainX, testX, trainy, testy)
            tempFit2,tempAcc2,tempCols2=Fit_KNN(tempPos2,trainX, testX, trainy, testy)
            z=0
            if tempFit1<tempFit2:
                 tempPos=tempPos1
                 tempFit=tempFit1
                 tempAcc=tempAcc1
                 tempCols=tempCols1
                 z=1
            else:
                tempPos=tempPos2
                tempFit=tempFit2
                tempAcc=tempAcc2
                tempCols=tempCols2
                z=2


            if(tempFit<popfit[i]):
                popPos[i]=tempPos
                popfit[i]=tempFit  
            if(tempFit<best_f):
                best_f=tempFit
                best_cols=tempCols
                best_x=tempPos 
                best_acc=tempAcc       
    return popPos,np.array(popfit),best_x,best_f, best_acc,best_cols


def PSO_DBSCAN(D,max_iter, Np,trainX, testX, trainy, testy , pop_pos_init, pop_fit_init,best_pop_init,best_fit_init,best_acc_init,best_cols_init,roound):
    

    w = 0.5                   # Intertial weight. In some variations, it is set to vary with iteration number.
    c1 = 2.0                  # Weight of searching based on the optima found by a particle
    c2 = 2.0                  # Weight of searching based on the optima found by the swarm
    v_fct = 1                 # Velocity adjust factor. Set to 1 for standard PSO.
    cov_cave=[best_fit_init]

 


    pbest_val = np.zeros(Np)            # Personal best fintess value. One pbest value per particle.
    gbest_val = np.zeros(max_iter)      # Global best fintess value. One gbest value per iteration (stored).

    best_acc=best_acc_init
    best_SF=best_cols_init
    tempAcc=np.zeros(Np)    
    tempSF= np.empty(Np, dtype=object)

    best_fit=best_fit_init


    pbest = np.zeros((Np,D))            # pbest solution
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

        print('PSO-DBSCAN,Itration : '+str(roound) +'-'+str(iter)+'  Fitness: '+str(best_fit)+'  Acc: '+str(best_acc)+
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

        if gbest_val[iter]<best_fit:
            best_fit=gbest_val[iter]
            best_acc=tempAcc[ind2]
            best_SF=tempSF[ind2]
        
        gbest_store[iter,:] = np.copy(gbest)              # store gbest solution
        
        pbest_val_avg_store[iter] = np.mean(pbest_val)
        fitness_avg_store[iter] = np.mean(fit)

        x,fit,gbest,best_fit,best_acc,best_SF=update_DBSCAN(x,fit,iter,trainX, testX, trainy, testy,gbest,best_fit, best_acc,best_SF)


        cov_cave.append(best_fit)
    return gbest, best_fit, cov_cave,best_acc,best_SF
