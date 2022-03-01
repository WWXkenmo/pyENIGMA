from collections import Counter
from random import choices
import multiprocessing
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from combat.pycombat import pycombat
from sklearn.linear_model import HuberRegressor
import scipy.stats as stats
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from scipy.optimize import minimize 
from scipy.optimize import LinearConstraint

def evaluate_func(parlist,X,y):
    epsilon = parlist[0]
    alpha = parlist[1]
    
    model = HuberRegressor(epsilon=epsilon,alpha=alpha)
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate model
    scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    return abs(scores.mean())
    
def optimize_para(parlist,X,y):
    
    #set constraint
    A = np.identity(2)
    lb = [1,0]
    ub = [np.inf,np.inf]
    cons = LinearConstraint(A, lb, ub, keep_feasible=False)
    
    #optimize
    result = minimize(evaluate_func,x0=parlist,args=(X, y), method="nelder-mead",constraints=cons)
    return result.x