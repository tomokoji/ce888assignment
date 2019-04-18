"""
===========================================================================
                      n n _ p a r a m e t e r s . p y
---------------------------------------------------------------------------
This code is to obtain parameters for neural networks (autoencoder, MLP).

Author          : Tomoko Ayakawa
Created on      : 17 April 2019
Last modified on: 18 April 2019
===========================================================================
"""
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../")
# import original libraries
from conf import myVariables as VAR

# -------------------------------------------------------------------------
# Define the parameters for the autorncoders/MLP. When skipped or an  
# invalid value is given, the default values that are defined in conf.py 
# are used.
# -------------------------------------------------------------------------
def msg(num_para):
    print("Define %d parameters for the autoencoder.\n" \
          "When skipped (push enter) or an invalid value is given, the " \
          "default value will be used.\n" % num_para)
    
def ae_type(i, j, default): # Autoencoder type
    try:
        mode=int(input("[Parameter %d/%d: Autoencoder type] "\
             "0:Normal or 1:Stacked (default=%d): " % (i, j, default)))
        if mode not in [0,1]: mode=default
    except:
        mode=default
    
    return mode

def ae_neurons(i, j, default): # Number of neurons of autoencoder
    layers=[str(k) for k in default]
    layers=input("[Parameter %d/%d: Number of neurons in each layer] "\
             "Integer separated by comma (default=%s): " % \
             (i, j, ",".join(layers)))
    
    mod_layers=[]
    try:
        for l in layers.split(","): mod_layers.append(int(l))
    except:
        mod_layers=default
    
    return mod_layers

def hidden_neurons(i, j, default): # Number of neurons in MLP hidden layer
    try:
        num_hid=int(input("[Parameter %d/%d: Number of hidden neurons]" \
                              "(default=%d): " % (i, j, default)))
        if num_hid<0: num_hid=default
    except:
            num_hid=default
    
    return num_hid
    
def activation(i, j, default, layer_type): # Activation function
    print("[Parameter %d/%d: Activation function%s] " \
          % (i, j, layer_type), end="")
    for k in VAR.act_list.keys(): 
        print("%d:%s " % (k, VAR.act_list[k]), end="")  
    act=input("(default=%s): " % VAR.act_list[default])
    try:
        act=VAR.act_list[int(act)]
    except:
        act=VAR.act_list[default]
    
    return act

def optimiser(i, j, default, def_lr, def_mom): # Optimiser
    from keras.optimizers import Adam, SGD
    print("[Parameter %d/%d: Optimiser] "  % (i, j), end="")
    for k in VAR.opt_list.keys(): 
        print("%d:%s " % (k, VAR.opt_list[k]), end="")
    opt=input("(default=%s): " % VAR.opt_list[default])
        
    try:
        opt=int(opt)
    except:
        opt=default
    
    if def_lr==-1: return VAR.opt_list[opt]
    if opt==0: #adam
        try:
            lr=float(input(" - Learning rate (default=%f): " % def_lr))
        except:
            lr=def_lr
        opt=Adam(lr=lr)
    else: # sdg
        try:
            lr=float(input(" - Learning rate (default=%f): " % def_lr))
        except:
            lr=def_lr
        try:
            momentum=float(input(" - Momentum (default=%f): " % def_mom))
        except:
            momentum=def_mom
        opt=SGD(lr=lr, momentum=momentum)
    
    return opt

def loss(i, j, default):  # Loss function
    print("[Parameter %d/%d: Loss function] " % (i, j), end="")
    for k in VAR.loss_list.keys(): 
        print("%d:%s " % (k, VAR.loss_list[k]), end="")
    loss=input("(default=%s): " % VAR.loss_list[default])
    try:
        loss=VAR.loss_list[int(loss)]
    except:
        loss=VAR.loss_list[default]
    
    return loss
    
def floatvalue(i, j, default, item): # Floatinf point values
    try:
        fl=float(input("[Parameter %d/%d: %s] 0<=rate<1 " \
                              "(default=%f): " % (i, j, item, default)))
        if (fl<0) or (fl>=1): fl=default
    except:
            fl=default
    
    return fl

def validation(i, j, default): # Dropout rate
    try:
        val=float(input("[Parameter %d/%d: Validation] 0<rate<1 " \
                              "(default=%f): " % (i, j, default)))
        if (val<=0) or (val>=1): val=default
    except:
            val=default
    
    return val
       
def integer(i, j, default, item): # Trainig epochs
    try:
        epochs=int(input("[Parameter %d/%d: %s] "\
                         "(default=%d): " % (i, j, item, default)))
        if epochs<0 : epochs=default
    except:
        epochs=default
    
    return epochs

def display(i, j, default, item_name): # Verbose, Tensor model summary
    try:
        display=int(input("[Parameter %d/%d: %s] 0:False or " \
                          "1:True (default=%d): " % \
                          (i, j, item_name, default)))
        if display not in [0,1]: display=default
 
    except:
        display=default
    
    return display