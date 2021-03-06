"""
===========================================================================
                      l o a d _ p h i s h i n g . p y
---------------------------------------------------------------------------
This code is read source files for phishing data and process the data
into the format which is appropriate for the following process.

Author          : Tomoko Ayakawa
Created on      : 5 February 2019
Submitted on    : 21 February 2019 (Assignment 1: Proposal)
Last modified on: 22 April 2019
===========================================================================
"""
import os.path, sys
import pandas as pd

sys.path.append("../")
from conf import myVariables as VAR

# -------------------------------------------------------------------------
# Read the data including column names, features and targets.
# -------------------------------------------------------------------------
def get_data(fname):
    if not(os.path.isfile(fname)):
        print ("[ ERROR ] The file <%s> does not exists." % fname)
        return None
    
    # read the file
    with open(fname, "r") as f:
        data = f.read()

    # split data into lines
    lines = data.split("\n")

    # split each lint into cells and obtain feature matrix
    col_names = []
    data = []
    
    for line in lines:
        if line != "":
            if line[0] == "@":
                cells = line.split(" ")
                if cells[0] == "@attribute":
                    col_names.append(cells[1])
            else:
                data.append(line.split(","))
    
    # replace the target column name from "Results" to "class"
    col_names = col_names[:-1] + ["Class"]
    
    # convert the data into DataFrame
    data_df = pd.DataFrame(data)
    
    # replace default column names with feature names
    data_df.columns = col_names
    
    # replace string representations into numeric values 
    data_df = data_df.astype(float)
    
    # split the data into features and targets
    features_df = data_df[data_df.columns[:-1]]
    targets_df = data_df[data_df.columns[-1]]

    # convert the features to dummy data
    ans=input("Convert nominal features to dummies?(y/n): ")
    if (ans=="y") or (ans=="Y"):
        features_df=pd.get_dummies(features_df.astype(str))
        col_names=features_df.columns
        data_df=features_df.join(pd.DataFrame(targets_df))

    return col_names, features_df, targets_df, data_df
# -------------------------------------------------------------------------
# Allow the programme to be ran from the command line.
# -------------------------------------------------------------------------
if __name__ == '__main__':
    # file = open_dialogue ()   
    col_names, features_df, targets_df, data_df = \
        get_data(VAR.phishing_data_file)
    print (data_df.head(5))
