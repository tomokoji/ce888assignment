"""
===========================================================================
                        a u t o e n c o d e r . p y
---------------------------------------------------------------------------
This code is to build and train an autoencoder for feature learning.

Author          : Tomoko Ayakawa
Created on      : 29 March 2019
Last modified on: 10 April 2019
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
# Build and train an autoencoder.
# Arguments:
#   - layers: number of neurons of the layers of autoencoder
#   - mode  : 0 for normal autoencoder, 1 for stacked autoencoder  
# -------------------------------------------------------------------------
def autoencoder(X, layers, mode, act=VAR.ae_act, opt=VAR.ae_opt, \
                loss=VAR.ae_loss, dropout=VAR.ae_dropout, \
                epochs=VAR.ae_epoch, verbose=1, summary_display=False): 
    
    from keras.models import Sequential, Model
    from keras.layers import Dense, Input, Dropout
    from keras.optimizers import Adam
    from keras.layers.advanced_activations import PReLU
    import warnings
    
    #ignore warnings
    warnings.filterwarnings("ignore")
    
    #create an input holder
    num_features=len(X[0])
    input_holder=Input(shape=(num_features,))
    
    #input_holder=Input(shape=(num_features,))
        
    #initialise the encoder variables
    layers.insert (0, num_features)
    
    # --------------------------------------------------------------------
    # normal autoencoder
    # --------------------------------------------------------------------
    if mode==0: 
        #initialise the decoder variables
        n=len(layers)
        layers_dec=[]
        
        for i in range(n-1, 0, -1): layers_dec.append(layers[i])     
        layers_dec.append(num_features)
        #act_funcs_dec=[act_funcs[-1]]*len(num_out_dec)
        
        print("num_outs:", layers, layers_dec)

        #build an encoder
        encode=input_holder
        for i in range (0, n-1):
            print("i=",i,"| output=", layers[i+1], ", input=",layers[i])
            
            encode=Dense(layers[i+1], input_dim=layers[i], \
                         activation=act)(encode)
            
            # add dropout
            if dropout!=0: 
                print("Add droppout")
                encode=Dropout(dropout)(encode)
                
        decode=encode
        for i in range (0, len(layers_dec)-1):
            print("i=",i,"| output=", layers_dec[i+1], \
                  ", input=",layers_dec[i])
            decode=Dense(layers_dec[i+1], input_dim=layers_dec[i], \
                            activation=act)(decode)
            
            # add dropout
            if dropout!=0:
                if i<len(layers_dec)-2:
                    print("Add droppout")
                    decode=Dropout(dropout)(decode)
                    
        autoencoder=Model(input=input_holder, output=decode)
        autoencoder.compile (optimizer=opt, loss=loss)
        
        if summary_display: print(autoencoder.summary())

        #train the autoencoder
        loss_hiss=autoencoder.fit(X, X, epochs=epochs, verbose=verbose)
        
        return encode, [loss_hiss]
    
    # --------------------------------------------------------------------
    # normal autoencoder
    # --------------------------------------------------------------------
    elif mode==1:
        tmp_holder=input_holder
        x=X
        encoders, loss_hist=[],[]
    
        print("layers:", layers)
        #train encoder layers
        for i in range(len(layers)-1):
            print("Training Layer %d/%d ..." % (i+1, len(layers)))
            print("i=",i,"| output=", layers[i+1], ", input=",layers[i])
            
            encode=Dense(layers[i+1], input_dim=layers[i], \
                           activation=act)(tmp_holder)
            if dropout!=0: 
                print("Add droppout")
                encode=Dropout(dropout)(encode)
                
            decode=Dense(layers[i], input_dim=layers[i+1], \
                           activation=act)(encode)
            encoder=Model(input=tmp_holder, output=decode)
            encoder.compile(optimizer=opt, loss=loss)
        
            # train a layer
            loss_his=encoder.fit(x, x, epochs=epochs, verbose=verbose)
            loss_hist.append(loss_his)
            
            # use output of the layer as next training input
            encoder=Model(input=tmp_holder, output=encode)
            x=encoder.predict(x)
            
            # update the input_holder
            tmp_holder=Input(shape=(layers[i+1],))
        
            # store the trainined layer in a list
            encoders.append(encoder)
            
        # connect traind encoder layers as 'encode'
        model=Sequential()
        for e in encoders:
            #print(e.summary())
            model.add(e)
            print(model.summary())
        
        if summary_display: print(model.summary())
            
        return model, loss_hist
    
# -------------------------------------------------------------------------
# Allow the programme to be ran from the command line.
# -------------------------------------------------------------------------
def plot_ae_loss_history(histories, mode, pic_file, save=True):
    # define grid
    n=len(histories)
    if n>2:
        col=2
        row=n//col+1
    else:
        row, col=1, n
        
    if mode==0: ae_type="Normal"
    else: ae_type="Stacked"
    
    fig=plt.figure(figsize=(col*6, row*4))
    
    for i in range(len(histories)):
        plt.subplot(row, col, i+1)
        plt.plot(histories[i].history["loss"])
        if mode==1: title="Layer %d " % (i+1)
        else: title=""
        plt.title("%s Autoencoder %sTraining Loss" % (ae_type, title))
    
    plt.show ()
    
    if save==True:
        fig.savefig("%s%s_%s_ae_loss.png" % \
                   (VAR.out_path, pic_file, ae_type), bbox_inches='tight')

# -------------------------------------------------------------------------
# Allow the programme to be ran from the command line.
# -------------------------------------------------------------------------
if __name__ == "__main__":
    X=np.random.rand(10,100)
    mode=1
    encode, histories=autoencoder(X, layers=[50, 30, 10], mode=mode, \
                act="relu", opt="adam", \
                loss="mse", dropout=0, \
                epochs=10, verbose=0, summary_display=False)
    
    plot_ae_loss_history(histories, mode, "test", save=True)