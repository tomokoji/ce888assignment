"""
===========================================================================
                        c o r r e l a t i o n . p y
---------------------------------------------------------------------------
This code displays and saves correlation of the features and the targets
in the data.

Author          : Tomoko Ayakawa
Created on      : 5 February 2019
Last modified on: 17 February 2019
===========================================================================
"""
import sys
import pandas as pd 
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../")
from conf import myVariables as VAR

# -------------------------------------------------------------------------
# Get correlation matrix.
# -------------------------------------------------------------------------
def correlation (data_df):
    cr_np = np.corrcoef (data_df.values.T)
    
    return cr_np

# -------------------------------------------------------------------------
# Show a heat map of correlation among features.
# -------------------------------------------------------------------------
def cr_heatmap(cr_np, pic_file, labels):
    size = len(labels) * 0.5
    fig = plt.figure(figsize = (size, size))
    sns.heatmap(cr_np, cbar = True, annot = True, square = True, 
                     fmt = '.2f', annot_kws = {'size': 8},
                     yticklabels = labels,
                     xticklabels = labels)
    plt.show ()
    
    fig.savefig("%s%s_crrel_hm.png" % (VAR.out_path, pic_file), \
                bbox_inches='tight')

# -------------------------------------------------------------------------
# Show a bar graph of correlation between features and targets.
# -------------------------------------------------------------------------
def cr_bar_graph (cr_np, pic_file, labels):
    # extract feature-target correlations from the matrix
    cr_np = cr_np[-1][:-1]
    
    # extract top correlated features
    indices = np.argsort(abs(cr_np))[::-1]
    cr_np = cr_np[indices [:VAR.crr_top]]
    labels = np.array(labels)[indices [:VAR.crr_top]]
    
    size_h = len (cr_np)*0.5
    size_v = size_h * 0.5
    
    fig = plt.figure (figsize = (size_h, size_v))
    axes = plt.gca()
    index = np.arange (len (cr_np))
    plt.bar(index, cr_np)
    plt.title ("Correlation with the Label (%s)" \
               % pic_file.split("_")[0], fontsize = 10)
    plt.xlabel("Features", fontsize = 10)
    plt.ylabel("Correlation", fontsize = 10)
    axes.set_ylim([-1,1])
    plt.xticks(index, labels, fontsize = 10, rotation = 90)
    plt.show ()
    
    fig.savefig("%s%s_crrel_bar.png" % (VAR.out_path, pic_file), \
                bbox_inches='tight')
    
# -------------------------------------------------------------------------
# Allow the programme to be ran from the command line.
# -------------------------------------------------------------------------
if __name__ == '__main__':
    # mock arguments
    data_df = pd.DataFrame([[1,2,3],[4,5,6],[9,8,7]])
    pic_file = "test"
    col_names = ["a", "b", "c"]
    
    cr_np = correlation (data_df)
    cr_heatmap (cr_np, pic_file, col_names)
    cr_bar_graph (cr_np, pic_file, col_names)
