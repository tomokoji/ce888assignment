"""
===========================================================================
                l o a d _ h u m a n _ a c t i v i t y . p y
---------------------------------------------------------------------------
This code is read source files for human activity data and process the data
into the format which is appropriate for the following process.

Author          : Tomoko Ayakawa
Created on      : 2 February 2019
Last modified on: 11 February 2019
===========================================================================
"""

# -------------------------------------------------------------------------
# Read feature names from the file
# -------------------------------------------------------------------------
def get_feature_names(fname):
    # read the file
    with open(fname, 'r') as f:
        data = f.read()

    # split the data into lines
    lines = data.split("\n")

    # split the lines into feature names
    feature_names = []
    for line in lines:
        cells = line.split(" ")
        if len(cells) > 1:
            feature_names.append ("%s_%s" % (cells[0], cells[1]))

    feature_names.append ("Class")

    return feature_names

# -------------------------------------------------------------------------
# Allow the programme to be ran from the command line.
# -------------------------------------------------------------------------
def read_all ():
    file_feature_name = "../data/human_activity/features.txt"
    feature_names = get_feature_names (file_feature_name)

    return feature_names
    
# -------------------------------------------------------------------------
# Allow the programme to be ran from the command line.
# -------------------------------------------------------------------------
if __name__ == '__main__':
    # file = open_dialogue ()
    
    read_all ()
