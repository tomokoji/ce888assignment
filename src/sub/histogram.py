"""
===========================================================================
                          h i s t o g r a m . p y
---------------------------------------------------------------------------
This code displays and saves histogram of data in each column of the data.

Author          : Tomoko Ayakawa
Created on      : 5 February 2019
Last modified on: 17 February 2019
===========================================================================
"""

import sys
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append("../")
from conf import myVariables as VAR

# -------------------------------------------------------------------------
# Show histograms of each column in the data
# -------------------------------------------------------------------------
def histogram(data_df, pic_file, mode):
    num_cols = len(data_df.columns)

    # convert DataFrame to numpy array
    #data = data_df.values

    # define figure grid
    if num_cols < 6: row, col = 1, num_cols
    else: row, col = num_cols//5+1, 5

    # create figure object
    plt.figure(figsize = (col*5, row*3))
    
    # plot data
    for i in range(num_cols):
        plt.subplot (row, col, i+1)
        plt.title (data_df.columns[i])
        sns.distplot(data_df[data_df.columns[i]].values, \
                     kde=False, rug=False)
    
    plt.savefig("%s%s_hist%s.png" % (VAR.out_path, pic_file, mode), \
                bbox_inches='tight')
    plt.show()

# -------------------------------------------------------------------------
# Allow the programme to be ran from the command line.
# -------------------------------------------------------------------------
if __name__ == '__main__':
    # mock arguments
    data_df = pd.DataFrame([[1,2,3],[4,5,6],[7,8,9]])
    pic_file = "test"
    
    histogram (data_df, pic_file, "")