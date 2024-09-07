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

    min_Pts=3
    if min_Pts<1:
        min_Pts=1  
    return eps,min_Pts
def DBSCAN_Update(popPos,popfit,it,trainX, testX, trainy, testy,best_x,best_f, best_acc,best_cols):

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

    return popPos,popfit,best_x,best_f, best_acc,best_cols

def GTO_DBSCAN(variables_no,max_iter, pop_size,trainX, testX, trainy, testy , pop_pos_init, pop_fit_init,best_pop_init,best_fit_init,best_acc_init,best_cols_init,roound):

    # Initialize Silverback
    Silverback = None
    Silverback_Score = np.inf

    best_acc = best_acc_init
    best_cols=best_cols_init

    convergence_curve = [best_fit_init]

    Pop_Fit =pop_fit_init
    Silverback_Score = best_fit_init
    Silverback = best_pop_init
    X=pop_pos_init
    GX = np.copy(X)
    lower_bound=-1000
    upper_bound=1000
    lb = np.ones(variables_no) * lower_bound
    ub = np.ones(variables_no) * upper_bound

    # Controlling parameters
    p = 0.03
    Beta = 3
    w = 0.8
    lower_bound=1000
    upper_bound=-1000
    # Main loop
    for It in range(max_iter-1):

        a = (np.cos(2 * np.random.rand()) + 1) * (1 - It / max_iter)
        C = a * (2 * np.random.rand() - 1)

        

        # Exploration
        for i in range(pop_size):
            if np.random.rand() < p:
                GX[i][:] = (ub - lb) * np.random.rand(variables_no) + lb
            else:
                if np.random.rand() >= 0.5:
                    Z = np.random.uniform(-a, a, variables_no)
                    H = Z * X[i][:]
                    GX[i][:] = (np.random.rand() - a) * X[np.random.randint(pop_size)][:] + C * H
                else:
                    r1 = np.random.randint(pop_size)
                    r2 = np.random.randint(pop_size)
                    GX[i][:] = X[i][:] - C * (C * (X[i][:] - GX[r1][ :]) + np.random.rand() * (X[i][:] - GX[r2][:]))

        GX = boundary_check(GX, lower_bound, upper_bound)
        # Group formation operation
        for i in range(pop_size):
            New_Fit,tempacc,tempcols = Fit_KNN(GX[i][:],trainX, testX, trainy, testy) 
            if New_Fit < Pop_Fit[i]:
                Pop_Fit[i] = New_Fit
                X[i][:] = GX[i][:]
            if New_Fit < Silverback_Score:
                Silverback_Score = New_Fit
                Silverback = GX[i][:]
                best_acc=tempacc
                best_cols=tempcols
        GX = boundary_check(GX, lower_bound, upper_bound)
        # Exploitation
        for i in range(pop_size):
            if a >= w:
                g = 2 ** C
                delta = (np.abs(np.mean(GX)) ** g) ** (1 / g)
                GX[i][:] = C * delta * (X[i][:] - Silverback) + X[i][:]
            else:
                if np.random.rand() >= 0.5:
                    h = np.random.randn(variables_no)
                else:
                    h = np.random.randn(1)
                r1 = np.random.rand()
                GX[i][:] = Silverback - (Silverback * (2 * r1 - 1) - X[i][:] * (2 * r1 - 1)) * (Beta * h)


        # Group formation operation
        for i in range(pop_size):
            New_Fit,tempacc,tempcols = Fit_KNN(GX[i][:],trainX, testX, trainy, testy)            
            if New_Fit < Pop_Fit[i]:
                Pop_Fit[i] = New_Fit
                X[i][:] = GX[i][:]
            if New_Fit < Silverback_Score:
                Silverback_Score = New_Fit
                Silverback = GX[i][:]
                best_acc=tempacc
                best_cols=tempcols

        convergence_curve.append(Silverback_Score)
        print('GTO_DBSCAN,Itration : '+str(roound) +'-'+str(It)+'  Fitness: '+str(Silverback_Score)+'  Acc: '+str(best_acc)+
              '  NumF: '+str(len(best_cols))+'  Features: '+str(best_cols))
        X,Pop_Fit,Silverback,Silverback_Score,best_acc,best_cols=DBSCAN_Update(X,Pop_Fit,It,trainX, testX, trainy, testy,Silverback,Silverback_Score, best_acc,best_cols)


    return Silverback, Silverback_Score, convergence_curve,best_acc,best_cols

def boundary_check(X, lb, ub):
    for i in range(X.shape[0]):
        FU = X[i, :] > ub
        FL = X[i, :] < lb
        X[i, :] = (X[i, :] * (~(FU | FL))) + ub * FU + lb * FL
    return X