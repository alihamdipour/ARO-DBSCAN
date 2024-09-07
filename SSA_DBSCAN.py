import numpy as np
from sklearn.cluster import DBSCAN
from torch import randperm
from matplotlib.pyplot import *
from pylab import *
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

    d_p=(math.dist(p2,p3))/temp_dis
    b=int((p_a[0]+len(M))/2)
    if d_p>=4:
        eps=kdis[p3[0]+b]
    else:
        eps=kdis[p3[0]+p_a[0]]
    
    min_Pts=math.ceil(d_p-0.5)
    if min_Pts<1:
        min_Pts=1  
    return eps,min_Pts
def DBSCAN_Update(popPos,popfit,it,trainX, testX, trainy, testy,best_x,best_f, best_acc,best_cols):

   
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
    return popPos,popfit,best_x,best_f, best_acc,best_cols
def SSA_DBSCAN(dim,Max_iter, N,trainX, testX, trainy, testy , pop_pos_init, pop_fit_init,best_pop_init,best_fit_init,best_acc_init,best_cols_init,roound):

    best_acc = best_acc_init
    best_cols=best_cols_init

    Fmax = 1.11
    Fmin = 0.5
    A = np.random.rand(N)
    r = np.random.rand(N)
    
    F = np.zeros(N)
    v = np.zeros((N, dim))
    
    x = pop_pos_init
    Convergence_curve = [best_fit_init]
    
    fitness = pop_fit_init
    update_ssa = np.random.randint(1, 4, size=fitness.shape)
    
    Gc = 1.9
    fmin = best_fit_init
    bestsol = best_pop_init
    
    iter = 0
    while iter < Max_iter-1:
        print('SSA-DBSCAN,Itration : '+str(roound) +'-'+str(iter)+'  Fitness: '+str(fmin)+'  Acc: '+str(best_acc)+
              '  NumF: '+str(len(best_cols))+'  Features: '+str(best_cols))

        for ii in range(N):
            if update_ssa[ii] == 1:
                F[ii] = Fmin + (Fmax - Fmin) * np.random.rand()
                v[ii][:] += F[ii] * Gc * (x[ii][:] - bestsol) * 1
                x[ii][:] += v[ii][:]
            elif update_ssa[ii] == 2:
                F[ii] = Fmin + (Fmax - Fmin) * np.random.rand()
                v[ii][:] += F[ii] * Gc * (x[ii][:] - bestsol) * 2
                x[ii][:] += v[ii][:]
            else:
                F[ii] = Fmin + (Fmax - Fmin) * np.random.rand()
                v[ii][:] += F[ii] * Gc * (x[ii][:] - bestsol) * 3
                x[ii][:] += v[ii][:]

            
            if np.random.rand() > r[ii]:
                eps = -1 + (1 - (-1)) * np.random.rand()
                x[ii][:] = bestsol + eps * np.mean(A)
            
            
            fitnessnew,tempacc,tempcols = Fit_KNN(x[ii][:],trainX, testX, trainy, testy)
            if fitnessnew <= fmin:
                bestsol = x[ii][:]
                fmin = fitnessnew
                best_acc=tempacc
                best_cols=tempcols
        
        Convergence_curve.append(fmin)
        x,F,bestsol,fmin,best_acc,best_cols=DBSCAN_Update(x,F,iter,trainX, testX, trainy, testy,bestsol,fmin, best_acc,best_cols)

        iter += 1

        
    return bestsol, fmin, Convergence_curve,best_acc,best_cols
