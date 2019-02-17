"""
===========================================================================
                l o a d _ h u m a n _ a c t i v i t y . p y
---------------------------------------------------------------------------
This code is read source files for human activity data and process the data
into the format which is appropriate for the following process.

Author          : Tomoko Ayakawa
Created on      : 2 February 2019
Last modified on: 17 February 2019
===========================================================================
"""
import os.path, sys
import pandas as pd

sys.path.append("../")
from conf import myVariables as VAR

# -------------------------------------------------------------------------
# Read feature names from the file
# -------------------------------------------------------------------------
def get_column_names(fname):
    if not(os.path.isfile(fname)):
        print ("[ ERROR ] The file <%s> does not exists." % fname)
        return None
    
    # read the file
    with open(fname, "r") as f:
        data = f.read()

    # split the data into lines
    lines = data.split("\n")

    # split the lines into feature names
    col_names = []
    for line in lines:
        cells = line.split(" ")
        if len(cells) > 1:
            col_names.append ("%s_%s" % (cells[0], cells[1]))

    col_names.append ("Class")

    return col_names

# -------------------------------------------------------------------------
# Read features from the file
# -------------------------------------------------------------------------
def get_features(fname):
    if not(os.path.isfile(fname)):
        print ("[ ERROR ] The file <%s> does not exists." % fname)
        return None
    
    # read the file
    with open(fname, "r") as f:
        data = f.read()

    # split data into lines
    lines = data.split("\n")
    
    # split each lint into cells and obtain feature matrix
    features = []
    for line in lines:
        cells = line.split(" ")
        features.append(value for value in cells if value != "")

    return pd.DataFrame(features)

# -------------------------------------------------------------------------
# Read features from the file
# -------------------------------------------------------------------------
def get_targets(fname):
    if not(os.path.isfile(fname)):
        print ("[ ERROR ] The file <%s> does not exists." % fname)
        return None
    
    # read the file
    with open(fname, 'r') as f:
        data = f.read()

    # split data into lines
    targets = data.split("\n")

    return pd.DataFrame(targets)

# -------------------------------------------------------------------------
# Allow the programme to be ran from the command line.
# -------------------------------------------------------------------------
if __name__ == '__main__':
    # file = open_dialogue ()
    
    get_column_names(VAR.human_acty_col_name_file)
    get_features(VAR.human_acty_feature_file)
    get_targets(VAR.human_acty_target_file)