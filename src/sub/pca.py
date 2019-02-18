"""
===========================================================================
                                 p c a . p y
---------------------------------------------------------------------------
This code analyses feature importance based on a Decision Tree classifier.

Author          : Tomoko Ayakawa
Created on      : 18 February 2019
Last modified on: 18 February 2019
===========================================================================
"""
import sys
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../")
from conf import myVariables as VAR

# -------------------------------------------------------------------------
# Set the predictors
# -------------------------------------------------------------------------
def pca(X, y, unique_labels, pic_file):
    from sklearn.decomposition import PCA as sklearnPCA
    from mpl_toolkits.mplot3d import Axes3D 
    
    # fit the data for PCA and plotting
    pca = sklearnPCA (n_components = 3) 
    X_pca = pd.DataFrame (pca.fit_transform (X)) 
    #y = y.reshape (1,len (y))[0]

    # colors and markers for each class
    colors = {0:"b", 1:"r", 2:"g", 3:"c", 4:"m", 5:"k"}
    markers = {0:"o", 1:"^", 2:"D", 3:"*", 4:"x", 5:"p"}
    num_labels = len(unique_labels)
    
    #Set a figure object
    #fig = plt.figure (figsize = (10, 10)) 
    #ax = fig.add_subplot (111, projection = "3d")
    
    print (X_pca)
    print (y)
    for i in range(num_labels):#ange(num_labels):
        print ("i=%d, label=%s" % (i,unique_labels[i]))
        print(y == unique_labels[i])
        print (X_pca[y == unique_labels[i]])

    #    ax.scatter(X_pca[y == labels[i]][0], X_pca[y == labels[i]][1], X_pca[y == labels[i]][2], 
    #               c = colors[i], marker = markers[i], label = labels[i])
    #ax.legend ()
    #plt.title (pic_file.split("_")[0])
    
    #plt.show()
    
    #fig.savefig("%s%s_pca.png" % \
    #            (VAR.out_path, pic_file), bbox_inches='tight')
    
# -------------------------------------------------------------------------
# Allow the programme to be ran from the command line.
# -------------------------------------------------------------------------
if __name__ == '__main__':
    import load_data as DATA
    data_id = 1
    col_names, features_df, targets_df, data_df, pic_file = \
        DATA.load_data(data_id=data_id)
    unique_labels = DATA.verify_data(data_df, targets_df)

    pca(features_df, targets_df, unique_labels, "test")                                   