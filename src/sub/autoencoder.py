"""
===========================================================================
                        a u t o e n c o d e r . p y
---------------------------------------------------------------------------
This code is to build and train an autoencoder for feature learning.

Author          : Tomoko Ayakawa
Created on      : 29 March 2019
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
import nn_parameters as PARA

# -------------------------------------------------------------------------
# Define the parameters for the autorncoders/MLP. When skipped or an  
# invalid value is given, the default values that are defined in conf.py 
# are used.
# -------------------------------------------------------------------------
def get_parameters(data_id):
    num_para=9
    PARA.msg(num_para)
    
    # Autoencoder type
    i=1
    mode=PARA.ae_type(i, num_para, VAR.ae_mode)

    # Number of neurons in each layer
    i+=1
    layers=PARA.ae_neurons(i, num_para, VAR.ae_layers[data_id])

    # Activation function
    i+=1
    act=PARA.activation(i, num_para, VAR.ae_act, "")

    # Optimiser
    i+=1
    opt=PARA.optimiser(i, num_para, VAR.ae_opt, VAR.ae_lr, VAR.ae_momentum)

    # Loss function
    i+=1
    loss=PARA.loss(i, num_para, VAR.ae_loss)
    
    # Dropout rate
    i+=1
    dropout=PARA.floatvalue(i, num_para, VAR.ae_dropout, "Dropout rate")
       
    # Trainig epochs
    i+=1
    epochs=PARA.integer(i, num_para, VAR.ae_epoch, "Training epochs")

    # Verbose
    i+=1
    verbose=PARA.display(i, num_para, VAR.ae_verbose, "Verbose")
    
    # summary_display
    i+=1
    summary_display=PARA.display(i, num_para, VAR.ae_summary_display, \
                                 "Summary display")
    
    print("\nParameters for the autoencoder are")
    print(" 1. Mode: %d(%s)" % (mode, VAR.ae_mode_list[mode]))
    print(" 2. Layers:", layers)
    print(" 3. Activation function:", act)
    print(" 4. Optimiser:", opt)
    print(" 5. Loss function:", loss)
    print(" 6. Dropout: %f" % dropout)
    print(" 7. Epochs: %d" % epochs)
    print(" 8. Verbose: %d(%s)" % (verbose, VAR.yesno[verbose]))
    print(" 9. Summary_display: %d(%s)" % (summary_display, \
          VAR.yesno[summary_display]))
    
    return layers, mode, act, opt, loss, dropout, epochs, verbose, \
           summary_display
        
# -------------------------------------------------------------------------
# Build and train an autoencoder.
# Arguments:
#   - layers: number of neurons of the layers of autoencoder
#   - mode  : 0 for normal autoencoder, 1 for stacked autoencoder  
# -------------------------------------------------------------------------
def autoencoder(X, layers, mode, act=VAR.ae_act, opt=VAR.ae_opt, \
                loss=VAR.ae_loss, dropout=VAR.ae_dropout, \
                epochs=VAR.ae_epoch, verbose=VAR.ae_verbose, \
                summary_display=VAR.ae_summary_display): 
    
    from keras.models import Sequential, Model
    from keras.layers import Dense, Input, Dropout
    import warnings
    
    #ignore warnings
    warnings.filterwarnings("ignore")
    
    #create an input holder
    num_features=len(X[0])
    input_holder=Input(shape=(num_features,))
    
    #input_holder=Input(shape=(num_features,))
        
    #initialise the encoder variables
    layers_copy=[num_features]
    for l in layers: layers_copy.append(l)
    
    # --------------------------------------------------------------------
    # basic autoencoder
    # --------------------------------------------------------------------
    if mode==0: 
        #initialise the decoder variables
        n=len(layers_copy)
        layers_dec=[]
        
        for i in range(n-1, 0, -1): layers_dec.append(layers_copy[i])     
        layers_dec.append(num_features)
        
        #build an encoder
        encoder=input_holder
        for i in range (0, n-1):            
            encoder=Dense(layers_copy[i+1], input_dim=layers_copy[i], \
                         activation=act)(encoder)
            
            # add dropout
            if dropout!=0: encoder=Dropout(dropout)(encoder)
                
        decoder=encoder
        for i in range (0, len(layers_dec)-1):
            decoder=Dense(layers_dec[i+1], input_dim=layers_dec[i], \
                            activation=act)(decoder)
            
            # add dropout
            if dropout!=0:
                if i<len(layers_dec)-2: decoder=Dropout(dropout)(decoder)
                    
        autoencoder=Model(input=input_holder, output=decoder)
        autoencoder.compile (optimizer=opt, loss=loss)
        
        #train the autoencoder
        loss_hiss=[autoencoder.fit(X, X, epochs=epochs, verbose=verbose)]
        
        # extract the encoder from the trained autoencoder
        encoder=Model(input=input_holder, output=encoder)
        
        if summary_display: print(autoencoder.summary())
        
        return encoder, loss_hiss
    
    # --------------------------------------------------------------------
    # stacked autoencoder
    # --------------------------------------------------------------------
    elif mode==1:
        tmp_holder=input_holder
        x=X
        stk_encoders, loss_hist=[],[]
    
        #train encoder layers
        for i in range(len(layers_copy)-1):
            print("Training Layer %d/%d ..." % (i+1, len(layers_copy)-1))
            
            encoder=Dense(layers_copy[i+1], input_dim=layers_copy[i], \
                           activation=act)(tmp_holder)
            if dropout!=0: encoder=Dropout(dropout)(encoder)
                
            decoder=Dense(layers_copy[i], input_dim=layers_copy[i+1], \
                           activation=act)(encoder)
            
            autoencoder=Model(input=tmp_holder, output=decoder)
            autoencoder.compile(optimizer=opt, loss=loss)
        
            # train a layer
            loss_his=autoencoder.fit(x, x, epochs=epochs, verbose=verbose)
            loss_hist.append(loss_his)
            
            # use output of the layer as next training input
            encoder=Model(input=tmp_holder, output=encoder)
            x=encoder.predict(x)
            
            # update the input_holder
            tmp_holder=Input(shape=(layers_copy[i+1],))
        
            # store the trainined layer in a list
            stk_encoders.append(encoder)
            
        # connect traind encoder layers as 'encode'
        encoder=Sequential()
        for e in stk_encoders: encoder.add(e)
        
        if summary_display==1: print(encoder.summary())
            
        return encoder, loss_hist
    
# -------------------------------------------------------------------------
# Plot the loss history and save it as a picture.
# -------------------------------------------------------------------------
def plot_ae_loss_history(histories, mode, pic_file):
    import matplotlib.ticker as ticker
    
    ans=input("Save training loss history as a picture? (y/n): ")
    if (ans=="y") or (ans=="Y"): save=True
    else: save=False 

    # define grid
    n=len(histories)
    if n>4:
        col=4
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
        plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator
               (integer=True))
    
    plt.show ()
    
    if save==True:
        fig.savefig("%s%s_%s_ae_loss.png" % \
                   (VAR.out_path, pic_file, ae_type), bbox_inches='tight')
     
# -------------------------------------------------------------------------
# Allow the programme to be ran from the command line.
# -------------------------------------------------------------------------
if __name__ == "__main__":
    X=np.random.rand(10,100)
    
    ae_layers, mode, act, opt, loss, dropout, epochs, verbose, \
           summary_display=get_parameters(data_id=0)
    
    ans=input("Continue? (y/n): ")
    
    if (ans=="y") or (ans=="Y"):
        encoder, histories=autoencoder(X, layers=ae_layers, mode=mode,\
                act=act, opt=opt, loss=loss, dropout=dropout, \
                epochs=epochs, verbose=verbose, \
                summary_display=summary_display)
    
    plot_ae_loss_history(histories, mode, "test")