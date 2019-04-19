"""
===========================================================================
                           c o m p a r i s o n . p y
---------------------------------------------------------------------------
This code is to compare the clasification results by an autoencoder to 
other classifiers.

Author          : Tomoko Ayakawa
Created on      : 19 April 2019
Last modified on: 19 April 2019
===========================================================================
"""
import sys
import pandas as pd
import numpy as np

sys.path.append("../")
# import original libraries
from conf import myVariables as VAR
import classifier as CLS
import pca as PCA

# -------------------------------------------------------------------------
# Create a table to compare the results of autoencoder with other 
# classifiers including Decision Tree, Naive Bayes, SVM and Neural Network
# -------------------------------------------------------------------------
def performance_comparison(pred_ae, X_tr, X_te, y_tr, y_te, \
                           pic_file, data_id):
    from sklearn.neural_network import MLPClassifier
    from sklearn import metrics
    
    ans=input("Display confusion matrix? (y/n): ")
    if (ans=="y") or (ans=="Y"): matrix=True
    else: matrix=False
        
    predictors=CLS.set_predictors()
    predictors.append(MLPClassifier())
    
    classifiers=["Autoencoder", "Decision Tree", \
                 "Naive Bayes", "SVM", "Neural Network"]
    
    predictions=[pred_ae]
    for p in predictors:
        p.fit(X_tr, y_tr)
        predictions.append(p.predict(X_te))
    
    results=[]
    for pred, c in zip(predictions, classifiers):
        acc=metrics.accuracy_score(y_te, pred)
        f=metrics.f1_score(y_te, pred, average='weighted')
        results.append([acc, f])
        
        if matrix==True:
            CLS.plot_confusion_matrix(y_te, pred, np.unique(y_te), \
                                      pic_file, c, c)
    
    results=pd.DataFrame(results, index=classifiers, \
                         columns=["Accuracy", "Wighted F-1"])
        
    print(results)
    
# -------------------------------------------------------------------------
# Compare 3D plot by PCA Autoencoder
# -------------------------------------------------------------------------
def plot_3D(X, X_cmp, y):
    PCA.pca(X_cmp, y, np.unique(y), \
                    "Compressed by autoencoder", PCA=False)
    PCA.pca(X, y, np.unique(y), "Compressed by PCA")