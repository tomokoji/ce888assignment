"""
===========================================================================
               f e a t u r e _ i m p o r t a n c e . p y
---------------------------------------------------------------------------
This code analyses feature importance based on a Decision Tree classifier.

Author          : Tomoko Ayakawa
Created on      : 5 February 2019
Last modified on: 17 February 2019
===========================================================================
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../")
from conf import myVariables as VAR
import classifier as CLS

# -------------------------------------------------------------------------
# Calculate and display feature importance.
# This method is written in reference to CE888 Lab 3 script.
# -------------------------------------------------------------------------
def feature_importance (clf, col_names, pic_file, title):
    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_], \
                 axis=0)
    indices = np.argsort(importances)[::-1]
    
    # select top features to display
    if len(col_names)-1 < VAR.imp_top:
        Y = importances
        Yerr = std
        X = range(len(Y))
    else:
        X = range(VAR.imp_top)
        Y = importances[indices[:VAR.imp_top]]
        Yerr = std[indices[:VAR.imp_top]]

    # Plot the feature importances of the forest
    fig = plt.figure()
    plt.bar(X, Y, color="b", yerr=Yerr, align="center")
    
    for x, y in zip(X, Y):
        plt.text(x, y, "%.5f" % y, ha='center', va='bottom')
        
    plt.xticks(X, np.array(col_names)[indices[:len(Y)]], rotation = 90)
    plt.xlim([-1, len(X)])
    fig.set_size_inches(len(Y),len(Y)/2)
    axes = plt.gca()
    axes.set_ylim([0,None])
    plt.title(title)
    
    plt.show()
    
    fig.savefig("%s%s_importance.png" % (VAR.out_path, pic_file), \
                     bbox_inches='tight')
    
# -------------------------------------------------------------------------
# Allow the programme to be ran from the command line.
# -------------------------------------------------------------------------
if __name__ == '__main__':
    from sklearn.datasets import load_iris
    data = load_iris()
    features = data.data
    target = data.target
    col_names = data.feature_names
    
    clf = CLS.train (features, target, 0)[-1]
    feature_importance(clf, col_names, "test", "test plot")
