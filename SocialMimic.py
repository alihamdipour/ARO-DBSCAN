import numpy as np
import pandas as pd
import random
import math,time,sys, os
from matplotlib import pyplot
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from config import *
#==================================================================
def sigmoid1(gamma):     #convert to probability
    if gamma < 0:
        return 1 - 1/(1 + math.exp(gamma))
    else:
        return 1/(1 + math.exp(-gamma))

def sigmoid1i(gamma):     #convert to probability
    gamma = -gamma
    if gamma < 0:
        return 1 - 1/(1 + math.exp(gamma))
    else:
        return 1/(1 + math.exp(-gamma))

def sigmoid2(gamma):
    gamma /= 2
    if gamma < 0:
        return 1 - 1/(1 + math.exp(gamma))
    else:
        return 1/(1 + math.exp(-gamma))
        
def sigmoid3(gamma):
    gamma /= 3
    if gamma < 0:
        return 1 - 1/(1 + math.exp(gamma))
    else:
        return 1/(1 + math.exp(-gamma))

def sigmoid4(gamma):
    gamma *= 2
    if gamma < 0:
        return 1 - 1/(1 + math.exp(gamma))
    else:
        return 1/(1 + math.exp(-gamma))


def Vfunction1(gamma):
    return abs(np.tanh(gamma))

def Vfunction2(gamma):
    val = (math.pi)**(0.5)
    val /= 2
    val *= gamma
    val = math.erf(val)
    return abs(val)

def Vfunction3(gamma):
    val = 1 + gamma*gamma
    val = math.sqrt(val)
    val = gamma/val
    return abs(val)

def Vfunction4(gamma):
    val=(math.pi/2)*gamma
    val=np.arctan(val)
    val=(2/math.pi)*val
    return abs(val)

def x1copy(gamma):
    s1 = abs(gamma)*0.5 + 1
    s1 = (-gamma)/s1 + 0.5
    return s1

def x2copy(gamma):
    s2 = abs(gamma - 1)*0.5 + 1
    s2 = (gamma - 1)/s2 + 0.5
    return s2





def toBinary(currAgent,prevAgent,dimension,trainX,testX,trainy,testy):
    # print("continuous",solution)
    # print(prevAgent)
    
    Xnew = np.zeros(np.shape(currAgent))
    for i in range(dimension):
        temp = Vfunction3(currAgent[i])

        random.seed(time.time()+i)
        # if temp > 0.5: # sfunction
        #     Xnew[i] = 1
        # else:
        #     Xnew[i] = 0
        if temp > 0.5: # vfunction
            Xnew[i] = 1 - prevAgent[i]
        else:
            Xnew[i] = prevAgent[i]
    return Xnew


def SocialMimic(dimension,maxIter,popSize,trainX, testX, trainy, testy, pop_pos_init, pop_fit_init,best_pop_init,best_fit_init,best_acc_init,best_cols_init,roound):

 
    x_axis = []
    y_axis = []
    population = pop_pos_init
    GBESTSOL =best_pop_init
    GBESTFIT = best_fit_init
    his_best_fit=[best_fit_init]
    best_acc=best_acc_init
    GBestAcc=best_acc_init
    bestCols=best_cols_init
    GBestCols=best_cols_init
    acclist=np.zeros(popSize).tolist()
    colsList=np.zeros(popSize).tolist()
    
    for currIter in range(1,maxIter):

        print('Socialmimic,Itration : '+str(roound)+'-'+str(currIter)+'  Fitness: '+str(GBESTFIT)+'  Acc: '+str(GBestAcc)+
              '  NumF: '+str(len(GBestCols))+'  Features: '+str(GBestCols))
        
        his_best_fit.append(GBESTFIT)
        newpop = np.zeros((popSize,dimension))

        x=np.shape(population)[0]
        fitList=np.zeros(x)
        for i in range(x):
            fitList[i],acclist[i],colsList[i]=Fit_KNN(population[i],trainX,testX,trainy,testy)  
            # fitList = allfit(population,trainX,testX,trainy,testy)
        if currIter==1:
            y_axis.append(min(fitList))
        else:
            y_axis.append(min(min(fitList),y_axis[len(y_axis)-1]))
        x_axis.append(currIter)
        bestInx = np.argmin(fitList)
        fitBest = min(fitList)
        Xbest = population[bestInx].copy()
        best_acc=acclist[bestInx]
        bestCols=colsList[bestInx]


        if fitBest<GBESTFIT:
            GBESTFIT = fitBest
            GBESTSOL = Xbest.copy()
            GBestAcc=best_acc
            GBestCols=bestCols
            print("gbest:",GBESTFIT,GBESTSOL.sum())

        for i in range(popSize):
            currFit = fitList[i]
            # print(currFit)
            difference = ( currFit - fitBest ) / currFit
            if difference == 0:
                random.seed(time.time())
                difference = random.uniform(0,1)
            newpop[i] = np.add(population[i],np.multiply(difference,population[i]))
            newpop[i] = toBinary(newpop[i],population[i],dimension,trainX,testX,trainy,testy)

        population = newpop.copy()

    return Xbest, GBESTFIT, his_best_fit,GBestAcc,GBestCols