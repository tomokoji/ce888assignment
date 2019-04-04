"""
===========================================================================
                        a u t o e n c o d e r . p y
---------------------------------------------------------------------------
This code is to build and train an autoencoder for feature extraction.

Author          : Tomoko Ayakawa
Created on      : 29 March 2019
Last modified on: 29 March 2019
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
def pre_processing(X, mode):   
    from sklearn.preprocessing import \
        MinMaxScaler, QuantileTransformer, StandartScaler
    
    # fit the scalar
    if mode==0: scl=MinMaxScaler()
    if mode==1: scl=QuantileTransformer(output_distribution = "normal")
    if mode==2: scl=StandardScaler()
    scl.fit (X)
    
    return (scl)
v
def define_encoder():
    from sklearn.model_selection import train_test_split
    
    test_size = 0.2
    X1_tr, X1_te, Y1_tr, Y1_te, ID1_tr, ID1_te = \
        train_test_split (X1, Y1, ID1, test_size = test_size)
    
    from keras.models import Sequential, Model
    from keras.layers import Dense, Input
    
    num_features = len (X1_tr[0])
    input_holder = Input (shape = (num_features,))
    
    num_outs = [55, 35, 15, 10]
    act_funcs = ["relu"] * len (num_outs) 

def train_encoder():
    from keras.optimizers import Adam
    from keras.layers.advanced_activations import PReLU
    
    #autoencoder = None
    num_outs.insert (0, num_features)
    tmp_holder = input_holder
    tr_data_ae = X_concatenated 
    encoders = []
    loss_hiss = []
    
    for i in range (len (act_funcs)):
        print ("Training Layer %d/%d ..." % (i + 1, len (act_funcs)))
        encode = Dense (num_outs[i + 1], input_dim = num_outs[i], activation = act_funcs [i])(tmp_holder)
        decode = Dense (num_outs[i], input_dim = num_outs[i + 1], activation = act_funcs [i])(encode)
        encoder = Model (input = tmp_holder, output = decode)
        encoder.compile (optimizer = "adam", loss = "mse")
    
        # train a layer
        ae_epochs = 20
        loss_his = encoder.fit (tr_data_ae, tr_data_ae, epochs = ae_epochs, verbose = 1)
        loss_hiss.append (loss_his)
        
        # use output of the layer as next training input
        encoder = Model (input = tmp_holder, output = encode)
        tr_data_ae = encoder.predict (tr_data_ae)
        
        # update the input_holder
        tmp_holder = Input (shape = (num_outs[i + 1],))
    
        # store the trainined layer in a list
        encoders.append (encoder)
        
    # connect traind encoder layers as 'encode'
    encoder = Sequential ()
    for e in encoders:
        encoder.add (e)
    
    
def log_loss_history (his, fname, key):
    loss = his.history[key]
    np_loss = np.array (loss)
    np.savetxt (fname, np_loss, delimiter = ",")
    
    print ("[  O K  ] The loss history is exported to <%s>" % fname)

    return ()

    # save the loss history to a text file
    for i in range (len (loss_hiss)):
        log_loss_stked = "loss_stackedautoencoder-%d_%s.txt" % (i + 1, now)
        log_loss_history (loss_hiss[i], log_loss_stked, "loss")
    
def train_discriminative_model():    
    #build a regression model by combining the encoder & MLP ANN
    model = Sequential()
    model.add (encoder)
    #model.add (Dense (5, activation = "sigmoid")) # hidden layer
    model.add (Dense (1, kernel_initializer = "normal", activation = "sigmoid"))
    model.compile (optimizer = "adam", loss = "mse")
    
    
    # In[ ]:
    
    
    # train the model (cross validation applied)
    rg_epochs = 100
    val_size = 0.2
    loss_his = model.fit (X1_tr, Y1_tr, validation_split = val_size, epochs = rg_epochs, verbose = 1)
    
    
    # In[ ]:
    
    
    # save the loss history to a text file
    log_loss_model = "loss_model_%s.txt" % now
    log_val_loss_model = "val_loss_model_%s.txt" % now
    
    np_loss = np.array (loss_his.history["loss"])
    np.savetxt (log_loss_model, np_loss, delimiter = ",")
    print ("[  O K  ] The loss history is exported to <%s>" % log_loss_model)
    
    np_loss = np.array (loss_his.history["val_loss"])
    np.savetxt (log_val_loss_model, np_loss, delimiter = ",")
    print ("[  O K  ] The loss history is exported to <%s>" % log_val_loss_model)
    
def prediction():    
    prediction_te = model.predict (X1_te)

    
def evaluation (actual, target):
    import math


    squared = [(x1 - x2)**2 for (x1, x2) in zip (actual, target)]
    mse = math.sqrt (sum (squared)/len (squared))

    print (mse)

    return (mse)
        
def scale_back (ID, P, Y): 
    # Scale back the outputs and targets
    Y = scl_Y.inverse_transform (Y)
    P = scl_Y.inverse_transform (np.array (P).reshape (-1, 1))
    
    #Y = scl_Y.inverse_transform (Y)
    #P = scl_Y.inverse_transform (P)

    # convert values from float to integer 
    ID = [int(i) for i in ID]
    P = [int(p) for p in P]
    Y = [int(y) for y in Y]

    return (ID, P, Y)

    ID, P_scbk, Y_scbk = scale_back (ID1_te, prediction_te, Y1_te)
    
    for i in range (len (X1_te)):
        print("ID %7s: Predicted = %s\tTarget = %s" % (ID[i], P_scbk[i], Y_scbk[i]))

# -------------------------------------------------------------------------
# Allow the programme to be ran from the command line.
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # import open source libraries
    import pandas as pd 
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt
    import sys
    
    # import original libraries
    sys.path.append("../")
    
    import assignment1_main as MAIN
    import load_data as DATA
    import histogram as HST
    import correlation as CRRL
    import pca as PCA
    import classifier as CLS
    import feature_importance as IMP
    
    data_list={0: "human activity", 1: "spam", 2: "phishing"}
    data_id = int(input(data_list))

    col_names, features_df, targets_df, data_df, pic_file = \
        DATA.load_data(data_id=data_id)
    unique_labels = DATA.verify_data(data_df, targets_df)
    
    pre_processing=pre_processing(data_df)