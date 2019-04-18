"""
===========================================================================
                        g r i d _ s e a r c h . p y
---------------------------------------------------------------------------
This code is to carry out grid-search for the optimal parameters of MLP.

Author          : Tomoko Ayakawa
Created on      : 18 April 2019
Last modified on: 18 April 2019
===========================================================================
"""
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../")
# import original libraries
from conf import myVariables as VAR
import nn_parameters as PARA

# -------------------------------------------------------------------------
# Define the parameters for the autorncoders/MLP. When skipped or an  
# invalid value is given, the default values that are defined in conf.py 
# are used.
# -------------------------------------------------------------------------
def get_parameters():
    num_para=8
    PARA.msg(num_para)

    # Activation function of a hidden layer
    i=1
    act=PARA.activation(i, num_para, VAR.h_act, "")

    # Number of hideen neurons
    i+=1
    h_num=PARA.integer(i, num_para, VAR.h_num, "Hidedn layer size")
    
    # Maximum iteration
    i+=1
    max_itr=PARA.integer(i, num_para, VAR.mlp_epoch, "Max iteration")
    
    # Learning rate
    i+=1
    lr=PARA.floatvalue(i, num_para, VAR.mlp_lr, "Learning rate")
    
    # Momentum
    i+=1
    mmt=PARA.floatvalue(i, num_para, VAR.mlp_momentum, "Momentum")
    
    # Regularization parameter
    i+=1
    alpha=PARA.floatvalue(i, num_para, VAR.alpha, \
                          "Regularization parameter")
    
    # Solver
    i+=1
    solver=PARA.optimiser(i, num_para, VAR.mlp_opt, -1, -1)
    
    # Grid split
    i+=1
    splits=PARA.integer(i, num_para, VAR.grid_splits, \
                        "Number of grid splits")
    
    
    print("\nParameters for grid search are")
    print(" 1. Activation function:", act)
    print(" 2. Hidden layer size:", h_num)
    print(" 3. Maximum iteration: %d" % max_itr)
    print(" 4. Learning rate: %f)" % lr)
    print(" 5. Momentum: %f)" % mmt)
    print(" 6. Regularization parameter: %f)" % alpha)
    print(" 7. Solver:", solver)
    print(" 8. Number of splits:", splits)
    
    return act, h_num, max_itr, lr, mmt, alpha, solver, splits
        
# -------------------------------------------------------------------------
# Set parameter grids.
# This code is originally developed by the author for the assignment of 
# CE802 which was submitted to the CSEE department in January 2019.           
# -------------------------------------------------------------------------
def parameter_grid(activation="relu", alpha=1, hidden_layer_sizes=(14,), 
                   max_iter=100, learning_rate_init=0.0001, momentum=0.9,
                   solver="adam"):
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import GridSearchCV, ShuffleSplit
    
    # create an MLP instance
    classifier=MLPClassifier(activation=activation, alpha=alpha, \
                   hidden_layer_sizes=hidden_layer_sizes, 
                   max_iter=max_iter, \
                   learning_rate_init=learning_rate_init,
                   solver=solver)
    
    # set parameter grids
    Hs=[(5, ), (10, ), (15, ), (20, ), (25, ), (30, )]
    ACTs=["tanh", "relu"]
    SLOs=["sgd", "adam"]
    ALPHAs=np.logspace(-5, 1, 7)
    ITRs=np.linspace(50, 1000, 100, dtype=int)
    ETAs=np.logspace(-5, -3, 3)
    MMTs=np.logspace(-5, -3, 3)

    grid_list=[]
    grid_list.append(dict(activation=ACTs, solver=SLOs))
    grid_list.append(dict(learning_rate_init=ETAs, momentum=MMTs))
    grid_list.append(dict(hidden_layer_sizes=Hs, alpha=ALPHAs))
    grid_list.append(dict(max_iter=ITRs))

    grid={0:"Activation function & Solver", \
          1:"Learning rate & Momentam", \
          2:"Hidden layer size & Regularization parameter", \
          3: "Max iteration"}
    print("Select the parameter to search - ", grid, end="")
    try:
        param_grid=grid_list[int(input(": "))]
    except:
        return 1, 1
    
    # set a classifier for grid search
    clf=GridSearchCV(estimator=classifier, param_grid=param_grid, \
                   n_jobs=-1, cv=ShuffleSplit())
    
    return param_grid, clf

# -------------------------------------------------------------------------
# Grid search.
# This code is originally developed by the author for the assignment of 
# CE802 which was submitted to the CSEE department in January 2019.           
# -------------------------------------------------------------------------
def grid_search(X, y, estimator, param_grid, grid_splits=10):
    from sklearn.model_selection import KFold
    from statistics import mode
    
    score, bests, lbs, means, modes=[], [], [], [], []
    k_fold=KFold(n_splits=grid_splits)

    grd=1
    for tr, te in k_fold.split(X):
        print("\rGrid Search Progress [%s%s]" % 
             (grd * "# ", (grid_splits-grd) * ". "), end = "")
        grd+=1
        best=[]

        estimator.fit(X[tr], y[tr])
        score.append(estimator.score(X[te], y[te]))

        for key in param_grid.keys():
            method=getattr(estimator.best_estimator_, key)
            best.append (method)
            if len(lbs)!=len(param_grid): lbs.append("%s" % key)
        best.append(score[-1])
        bests.append(best)

    lbs.append("Accuracy")

    # append mean and mode
    for k in range (len(bests[0])):
        avgs=(np.array(bests)[:, k]).tolist()
        try:    modes.append(mode (avgs))
        except: modes.append("N/A")

        try:    means.append(np.mean ([float (avg) for avg in avgs]))
        except: means.append("N/A")

    index=["Fold-%2d" % fold for fold in(range(1, grd))]
    for add1, add2 in zip([["---"]*(len(param_grid)+1), means, modes], 
                           ["---", "mean", "modes"]):
        bests.append(add1)
        index.append(add2)
    
    # display best scores
    results=pd.DataFrame(bests, index=index, columns=lbs)
    print("\n", results)
    
    return results
     
# -------------------------------------------------------------------------
# Allow the programme to be ran from the command line.
# -------------------------------------------------------------------------
if __name__ == "__main__":
    X=np.random.rand(50,100)
    y=np.array([0,1,2,3,4]*10)
    
    act, h_num, max_itr, lr, mmt, alpha, solver, splits=get_parameters()
    
    ans=input("Continue? (y/n): ")
    
    if (ans=="y") or (ans=="Y"):
        param_grid, clf=parameter_grid(activation=act, alpha=alpha, \
                       hidden_layer_sizes=(h_num,), 
                       max_iter=max_itr, learning_rate_init=lr, 
                       momentum=mmt, solver=solver)
        if param_grid!=1:
            results=grid_search(X, y, clf, param_grid, grid_splits=splits)