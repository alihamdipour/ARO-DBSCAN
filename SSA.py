import numpy as np
from sklearn.cluster import DBSCAN
from torch import randperm
from matplotlib.pyplot import *
from pylab import *
from config import *

def SSA(dim,Max_iter, N,trainX, testX, trainy, testy , pop_pos_init, pop_fit_init,best_pop_init,best_fit_init,best_acc_init,best_cols_init,roound):

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
        print('SSA,Itration : '+str(roound) +'-'+str(iter)+'  Fitness: '+str(fmin)+'  Acc: '+str(best_acc)+
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
        iter += 1
        
    return bestsol, fmin, Convergence_curve,best_acc,best_cols
