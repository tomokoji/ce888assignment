"""
===========================================================================
                                 p c a . p y
---------------------------------------------------------------------------
This code carries out PCA, plot the data sets in 3D and display information
(variance) provided by principal components.
This code is written in reference to jupyter notebook for CE888 Lab 6. 

Author          : Tomoko Ayakawa
Created on      : 18 February 2019
Last modified on: 19 February 2019
===========================================================================
"""
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as sklearnPCA

sys.path.append("../")
from conf import myVariables as VAR

# -------------------------------------------------------------------------
# Set the predictors.
# -------------------------------------------------------------------------
def pca(X, y, labels, pic_file, PCA=True):
    from mpl_toolkits.mplot3d import Axes3D 
    
    # fit the data for PCA and plotting
    # when PCA=False, plot the original data into 3D
    pca = sklearnPCA (n_components = 3) 
    if PCA==True: X_pca = pd.DataFrame (pca.fit_transform (X)) 
    else: X_pca=X

    # colors and markers for each class
    colors = {0:"b", 1:"r", 2:"g", 3:"c", 4:"m", 5:"k"}
    markers = {0:"o", 1:"^", 2:"D", 3:"*", 4:"x", 5:"p"}
    num_labels = len(labels)
    
    #Set a figure object
    ans=input("Save 3D plot of samples as a picture? (y/n): ")
    if (ans=="y") or (ans=="Y"): save=True
    else: save=False
    
    fig = plt.figure (figsize = (10, 10)) 
    ax = fig.add_subplot (111, projection = "3d")
    
    for i in range(num_labels):
        X_in_cls=X_pca[y == labels[i]]
        ax.scatter(X_in_cls[0], X_in_cls[1], X_in_cls[2], \
                   c = colors[i], marker = markers[i], label = labels[i])
    ax.legend ()
    plt.title (pic_file.split("_")[0])
    
    plt.show()
    
    if save==True:
        fig.savefig("%s%s_pca.png" % (VAR.out_path, pic_file), \
                bbox_inches="tight")

def variance(X, pic_file):
    pca = sklearnPCA()
    X_pca = pca.fit_transform(X)
    pca.explained_variance_ratio_
    variance = pca.explained_variance_ratio_
    n = len(variance)

    fig = plt.figure () 
    plt.bar(range(1, n+1), variance, alpha=0.5, align="center")
    plt.step(range(1, n+1), np.cumsum(variance), where="mid")
    plt.ylabel("Explained Variance Ratio")
    plt.xlabel("Principal Components")
    plt.title (pic_file.split("_")[0])
    
    plt.show()

    fig.savefig("%s%s_pca_variance.png" % \
                (VAR.out_path, pic_file), bbox_inches="tight")

# -------------------------------------------------------------------------
# Allow the programme to be ran from the command line.
# -------------------------------------------------------------------------
if __name__ == "__main__":
    import load_data as DATA
    data_id = 2
    col_names, features_df, targets_df, data_df, pic_file = \
        DATA.load_data(data_id=data_id)
    unique_labels = DATA.verify_data(data_df, targets_df)

    pca(features_df, targets_df, unique_labels, "test")
    variance(features_df, pic_file)
