"""
==========================================================================
                     a s s i g n m e n t 1. p y
---------------------------------------------------------------------------
This code is written as part of requirments of CE888 Data Science and 
Decision Making for Task 1 of Project 1.

Author          : Tomoko Ayakawa
Created on      : 2 February 2019
Last modified on: 2 February 2019
===========================================================================
"""

import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------------
def histogram(df_data):
    fig = plt.figure()

    i = 1
    for header in df_data.columns:
        print (header)
        plt.subplots(2,1,i+1)
        
        i += 1
        
        #sns.distplot(df_data[header], bins=20, kde=False, rug=True, \
        #         label=header).get_figure()

        #axes = plt.gca()
        #axes.set_xlabel('Value') 
        #axes.set_ylabel('Count')
        #axes.legend ()
    
        
        #plt.title ()
        #sns.distplot(df_data[:, i]);
    
        #plt.show ()
        #plt.savefig ('vehicles_plot.png')
        
    
# -------------------------------------------------------------------------
# Open a dialog to select a file to open.
# -------------------------------------------------------------------------
def open_dialogue():
    import os, tkinter, tkinter.filedialog, tkinter.messagebox
    
    root = tkinter.Tk()
    root.withdraw ()
    fTyp = [('','*.csv')]
    iDir = os.path.abspath (os.path.dirname(__file__))

    title = 'Select a file to open',
    file = tkinter.filedialog.askopenfilename \
           (filetypes = fTyp, initialdir = iDir, title = title)
    
    return (file)

# -------------------------------------------------------------------------
# Allow the programme to be ran from the command line.
# -------------------------------------------------------------------------
if __name__ == '__main__':
    # file = open_dialogue ()
    file = './sample.csv'
    
    if file != "":
        df_data = pd.read_csv(file)
        
        histogram (df_data)
        