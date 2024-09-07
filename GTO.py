import numpy as np
from sklearn.cluster import DBSCAN
from torch import randperm
from matplotlib.pyplot import *
from pylab import *
from config import *

def GTO(variables_no,max_iter, pop_size,trainX, testX, trainy, testy , pop_pos_init, pop_fit_init,best_pop_init,best_fit_init,best_acc_init,best_cols_init,roound):

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
    lower_bound=1000
    upper_bound=-1000
    ub=upper_bound
    lb=lower_bound

    # Controlling parameters
    p = 0.03
    Beta = 3
    w = 0.8

    # Main loop
    for It in range(max_iter-1):
        a = (np.cos(2 * np.random.rand()) + 1) * (1 - It / max_iter)
        C = a * (2 * np.random.rand() - 1)

        # Exploration
        for i in range(pop_size):
            if np.random.rand() < p:
                GX[i, :] = (ub - lb) * np.random.rand(variables_no) + lb
            else:
                if np.random.rand() >= 0.5:
                    Z = np.random.uniform(-a, a, variables_no)
                    H = Z * X[i, :]
                    GX[i, :] = (np.random.rand() - a) * X[np.random.randint(pop_size), :] + C * H
                else:
                    r1 = np.random.randint(pop_size)
                    r2 = np.random.randint(pop_size)
                    GX[i, :] = X[i, :] - C * (C * (X[i, :] - GX[r1, :]) + np.random.rand() * (X[i, :] - GX[r2, :]))

        
        GX = boundary_check(GX, lower_bound, upper_bound)

        # Group formation operation
        for i in range(pop_size):
            New_Fit,tempacc,tempcols = Fit_KNN(GX[i, :],trainX, testX, trainy, testy) 
            if New_Fit < Pop_Fit[i]:
                Pop_Fit[i] = New_Fit
                X[i, :] = GX[i, :]
            if New_Fit < Silverback_Score:
                Silverback_Score = New_Fit
                Silverback = GX[i, :]
                best_acc=tempacc
                best_cols=tempcols
        GX = boundary_check(GX, lower_bound, upper_bound)

        # Exploitation
        for i in range(pop_size):
            if a >= w:
                g = 2 ** C
                delta = (np.abs(np.mean(GX)) ** g) ** (1 / g)
                GX[i, :] = C * delta * (X[i, :] - Silverback) + X[i, :]
            else:
                if np.random.rand() >= 0.5:
                    h = np.random.randn(variables_no)
                else:
                    h = np.random.randn(1)
                r1 = np.random.rand()
                GX[i, :] = Silverback - (Silverback * (2 * r1 - 1) - X[i, :] * (2 * r1 - 1)) * (Beta * h)


        # Group formation operation
        for i in range(pop_size):
            New_Fit,tempacc,tempcols = Fit_KNN(GX[i, :],trainX, testX, trainy, testy)            
            if New_Fit < Pop_Fit[i]:
                Pop_Fit[i] = New_Fit
                X[i, :] = GX[i, :]
            if New_Fit < Silverback_Score:
                Silverback_Score = New_Fit
                Silverback = GX[i, :]
                best_acc=tempacc
                best_cols=tempcols

        convergence_curve.append(Silverback_Score)
        print('GTO,Itration : '+str(roound) +'-'+str(It)+'  Fitness: '+str(Silverback_Score)+'  Acc: '+str(best_acc)+
              '  NumF: '+str(len(best_cols))+'  Features: '+str(best_cols))

    return Silverback, Silverback_Score, convergence_curve,best_acc,best_cols

def boundary_check(X, lb, ub):
    for i in range(X.shape[0]):
        FU = X[i, :] > ub
        FL = X[i, :] < lb
        X[i, :] = (X[i, :] * (~(FU | FL))) + ub * FU + lb * FL
    return X