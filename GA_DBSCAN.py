# genetic algorithm search of the one max optimization problem
from numpy.random import randint
from numpy.random import rand
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
    temp_dis=(math.dist(p1,p3)) 
    if temp_dis==0:
        temp_dis=1
    d_p=(math.dist(p2,p3))/ temp_dis
    b=int((p_a[0]+len(M))/2)
    if d_p>=4:
        eps=kdis[p3[0]+b]
    else:
        eps=kdis[p3[0]+p_a[0]]
    # mn=math.ceil((100-it)/10)+1

    # eps/=1.5
    # if it%5==0:
    #     plot(kdis)
    #     line1=[]
    #     line2=[]
    #     line3=[]
    #     for x in range(0,len(kdis)):
    #         line1.append(A1*x+B1)
    #         line2.append(A2*x+B2)
    #         line3.append(A3*x+B3)        
    #     plot(line1)
    #     plot(line2)
    #     plot(line3)
    #     plt.ylim([0,kdis[-1]])
    #     if d_p>=4:
    #         plt.plot(p3[0]+b, p3[1]+b, marker="*", markersize=5, markeredgecolor="green")
    #     else:
    #         plt.plot(p3[0]+p_a[0], p3[1]+p_a[1], marker="o", markersize=5, markeredgecolor="green")
        
    #     if d_p>=4:
    #         plt.plot(p3[0]-b, p3[1]-b, marker="*", markersize=5, markeredgecolor="red")
    #     else:
    #         plt.plot(p3[0]-p_a[0], p3[1]-p_a[1], marker="o", markersize=5, markeredgecolor="red")
    #     show()
    min_Pts=math.ceil(d_p-0.5)
    if min_Pts<1:
        min_Pts=1  
    return eps,min_Pts

def update_DBSCAN(popPos,popfit,it,trainX, testX, trainy, testy,best_x,best_f, best_acc,best_cols):

    popfit,popPos  = (list(t) for t in zip(*sorted(zip(popfit, popPos) ,key=lambda x:x[0])))
    radus,min_Pts=params(popPos,len(popPos),it)
    # print('raduse= '+str(radus)+'   minPts= '+str(min_Pts) )
    db = DBSCAN(eps=radus+0.00000001, min_samples=min_Pts).fit(popPos)
    labels = db.labels_
    # print(labels)

    for i in range(1,len(popPos)):         
        indexs = [m for m ,e in enumerate(labels) if e == labels[i]]
        if  i!=indexs[0]  and len(indexs)>2 and labels[i]!=-1 :
            tempPos1=deepcopy(popPos[i])   
            tempPos2=deepcopy(popPos[i])            
         
            tempPos1=np.bitwise_and(tempPos1,np.bitwise_xor(popPos[indexs[0]],popPos[indexs[1]]))
            tempPos2=np.bitwise_and(tempPos2,np.bitwise_or(popPos[indexs[0]],popPos[indexs[1]]))
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
                # if z==1:
                #     print('*************************************************************************')
                # if z==2:
                #     print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            if(tempFit<best_f):
                # print('TTTTTTTTTTTTTTTTTTTTTTTT')
                best_f=tempFit
                best_cols=tempCols
                best_x=tempPos 
                best_acc=tempAcc     
    for i in range(len(popPos)) :
        if type(popPos[i])!=list:
            popPos[i]= popPos[i].tolist()
            
    return popPos,popfit,best_x,best_f, best_acc,best_cols
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
def GA_DBSCAN(n_bits,n_iter, n_pop,trainX, testX, trainy, testy , pop_pos_init, pop_fit_init,best_pop_init,best_fit_init,best_acc_init,best_cols_init,roound):


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
        print('GA-DBSCAN,Itration : '+str(roound) +'-'+str(gen)+'  Fitness: '+str(best_eval)+'  Acc: '+str(bestAcc)+
              '  NumF: '+str(len(bestFeatures))+'  Features: '+str(bestFeatures))
        
        pop,scores,best,best_eval,bestAcc,bestFeatures=update_DBSCAN(pop,scores,gen,trainX, testX, trainy, testy,best,best_eval, bestAcc,bestFeatures)

    return best, best_eval, con_cave,bestAcc,bestFeatures