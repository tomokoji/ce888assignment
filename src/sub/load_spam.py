"""
===========================================================================
                      l o a d _ s p a m . p y
---------------------------------------------------------------------------
This code is read source files for spam data and process the data
into the format which is appropriate for the following process.

Author          : Tomoko Ayakawa
Created on      : 5 February 2019
Last modified on: 17 February 2019
===========================================================================
"""

import sys
import pandas as pd

sys.path.append("../")
from conf import myVariables as VAR

# -------------------------------------------------------------------------
# Read the column names.
# -------------------------------------------------------------------------
def get_col_names(fname):
    with open(fname, 'r') as f:
        data = f.read()
    
    # split data into lines
    lines = data.split("\n")
    
    # obtain feature names
    col_names = []
    
    for line in lines:
        cells = line.split(":")
        if len(cells) == 2: 
            col_names.append (cells[0])
    
    col_names.append ("Class")

    return col_names

# -------------------------------------------------------------------------
# Read the column names.
# -------------------------------------------------------------------------
def get_data(fname, col_names):
    with open(fname, 'r') as f:
        data = f.read()
    
    # split data into lines
    lines = data.split("\n")
    
    # split each lint into cells and obtain feature matrix
    data = []
    
    for line in lines:
        cells = line.split(",")
        data.append(cells)
                
    # convert the data into DataFrame
    data_df = pd.DataFrame(data)
    
    # replace default column names with feature names
    data_df.columns = col_names
    
    # replace string representations into numeric values
    data_df = data_df.convert_objects(convert_numeric=True)
    
    # separate the features and the targets
    features_df=data_df[data_df.columns[:-1]]
    targets_df=data_df[data_df.columns[-1:]]

    return features_df, targets_df, data_df
# -------------------------------------------------------------------------
# Allow the programme to be ran from the command line.
# -------------------------------------------------------------------------
if __name__ == '__main__':
    # file = open_dialogue ()   
    col_names = get_col_names(VAR.spam_col_name_file)
    features_df, targets_df, data_df = \
            get_data(VAR.spam_data_file, col_names)
    print (data_df.head(5))
