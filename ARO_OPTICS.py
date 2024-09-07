import numpy as np
from sklearn.cluster import OPTICS
from torch import randperm
from matplotlib.pyplot import *
from pylab import *
import math
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
        min_Pts=2
    return eps,min_Pts

def update_DBSCAN(popPos,popfit,it,trainX, testX, trainy, testy,best_x,best_f, best_acc,best_cols):

    popfit,popPos  = (list(t) for t in zip(*sorted(zip(popfit, popPos) ,key=lambda x:x[0])))
    radus,min_Pts=params(popPos,len(popPos),it)
    # print('raduse= '+str(radus)+'   minPts= '+str(min_Pts) )
    db = OPTICS(eps=radus+0.00000001, min_samples=3).fit(popPos)
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

def ARO_OPTICS(dim,max_it, npop,trainX, testX, trainy, testy , pop_pos_init, pop_fit_init,best_pop_init,best_fit_init,best_acc_init,best_cols_init,roound):

    best_acc = best_acc_init
    best_cols=best_cols_init
    pop_pos = pop_pos_init
    pop_fit = pop_fit_init
    best_f = best_fit_init
    best_x = best_pop_init
    
    his_best_fit = [best_fit_init]

    for it in range(max_it-1):
        print('ARO-OPTICS,Itration : '+str(roound) +'-'+str(it)+'  Fitness: '+str(best_f)+'  Acc: '+str(best_acc)+
              '  NumF: '+str(len(best_cols))+'  Features: '+str(best_cols))

        direct1=np.zeros((npop, dim))
        direct2=np.zeros((npop, dim))
        theta = 2 * (1 - (it+1) / max_it)
        for i in range(npop):
            L = (np.e - np.exp((((it+1) - 1) / max_it) ** 2)) * (np.sin(2 * np.pi * np.random.rand())) # Eq.(3)
            rd = np.floor(np.random.rand() * (dim))
            rand_dim = randperm(dim)
            direct1[i, rand_dim[:int(rd)]] = 1
            c = direct1[i,:]  #Eq.(4)
            R = L * c # Eq.(2)
            A = 2 * np.log(1 / np.random.rand()) * theta #Eq.(15)
            if A>1:
               K=np.r_[0:i,i+1:npop]
               RandInd=(K[np.random.randint(0,npop-1)])
               newPopPos = pop_pos[RandInd][:] + R * (pop_pos[i][:] - pop_pos[RandInd][:])+round(0.5 * (0.05 +np.random.rand())) * np.random.randn() # Eq.(1)
            else:
                ttt=int(np.floor(np.random.rand() * dim))
                direct2[i, ttt] = 1
                gr = direct2[i,:] #Eq.(12)
                H = ((max_it - (it+1) + 1) / max_it) * np.random.randn() # % Eq.(8)
                b = pop_pos[i][:]+H * gr * pop_pos[i][:] # % Eq.(13)
                newPopPos = pop_pos[i][:]+ R* (np.random.rand() * b - pop_pos[i][:]) #Eq.(11)

            newPopFit,tempacc,tempcols = Fit_KNN(newPopPos,trainX, testX, trainy, testy)
            if newPopFit < pop_fit[i]:
               pop_fit[i] = newPopFit
               pop_pos[i][:] = newPopPos
   


            if pop_fit[i] < best_f:
               best_f = pop_fit[i]
               best_x = pop_pos[i][:]
               best_acc=tempacc
               best_cols=tempcols
        pop_pos,pop_fit,best_x,best_f,best_acc,best_cols=update_DBSCAN(pop_pos,pop_fit,it,trainX, testX, trainy, testy,best_x,best_f, best_acc,best_cols)
        his_best_fit.append(best_f)
    return best_x, best_f, his_best_fit,best_acc,best_cols
