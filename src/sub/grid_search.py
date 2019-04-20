"""
===========================================================================
                        g r i d _ s e a r c h . p y
---------------------------------------------------------------------------
This code is to carry out grid-search for the optimal parameters of MLP.

Author          : Tomoko Ayakawa
Created on      : 18 April 2019
Last modified on: 19 April 2019
===========================================================================
"""
import sys
import pandas as pd
import numpy as np

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
    num_para=6
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
    print(" 4. Learning rate: %f" % lr)
    print(" 5. Solver:", solver)
    print(" 6. Number of splits:", splits)
    
    return act, h_num, max_itr, lr, solver, splits
        
# -------------------------------------------------------------------------
# Set parameter grids.
# This code is originally developed by the author for the assignment of 
# CE802 which was submitted to the CSEE department in January 2019.           
# -------------------------------------------------------------------------
def parameter_grid(activation="relu", hidden_layer_sizes=(20,), 
                   max_iter=1000, learning_rate_init=0.001, solver="adam"):
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import GridSearchCV, ShuffleSplit
    
    # create an MLP instance
    classifier=MLPClassifier(activation=activation, \
               hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, 
               solver=solver, learning_rate_init=learning_rate_init)
                   
    
    # set parameter grids
    Hs=[(5, ), (10, ), (15, ), (20, ), (25, ), (30, )]
    ACTs=["tanh", "relu"]
    SLOs=["sgd", "adam"]
    ITRs=np.linspace(10, 1000, 100, dtype=int)
    ETAs=np.logspace(-5, -1, 5)

    grid_list=[]
    grid_list.append(dict(activation=ACTs))
    grid_list.append(dict(solver=SLOs))
    grid_list.append(dict(learning_rate_init=ETAs))
    grid_list.append(dict(hidden_layer_sizes=Hs))
    grid_list.append(dict(max_iter=ITRs))

    grid={0:"Activation function", \
          1:"Solver", \
          2:"Learning rate", \
          3:"Hidden layer size", \
          4:"Max iteration"}
    print("Select the parameters to search (separated by comma) - ", \
          grid, end="")
    ans=input(": ").split(",")
    param_grid=0

    try:
        for g in ans:
            g=int(g)
            if param_grid==0: param_grid=grid_list[g]
            else: param_grid.update(grid_list[g])
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
    from datetime import datetime
    
    
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
        try:    modes.append(mode(avgs))
        except: modes.append("N/A")

        try:    means.append(np.mean ([float(avg) for avg in avgs]))
        except: means.append("N/A")

    index=["Fold-%2d" % fold for fold in(range(1, grd))]
    for add1, add2 in zip([["---"]*(len(param_grid)+1), means, modes], 
                           ["---", "mean", "modes"]):
        bests.append(add1)
        index.append(add2)
    
    # display best scores
    results=pd.DataFrame(bests, index=index, columns=lbs)
    print("\n", results)
    
    ans=input("Save the results to a CSV file? (y/n): ")
    if (ans!="y") or (ans!="Y"):
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        results.to_csv("%sGridSearch_%s.csv" % (VAR.out_path, now), \
                       header=lbs, index=index, sep=",", mode="a")
    
    return results
     
# -------------------------------------------------------------------------
# Allow the programme to be ran from the command line.
# -------------------------------------------------------------------------
if __name__ == "__main__":
    X=np.random.rand(50,100)
    y=np.array([0,1,2,3,4]*10)
    
    act, h_num, max_itr, lr, solver, splits=get_parameters()
    
    ans=input("Continue? (y/n): ")
    
    if (ans=="y") or (ans=="Y"):
        param_grid, clf=parameter_grid(activation=act, \
                        hidden_layer_sizes=(h_num,), max_iter=max_itr, \
                        learning_rate_init=lr, solver=solver)

        if param_grid!=1:
            results=grid_search(X, y, clf, param_grid, grid_splits=splits)