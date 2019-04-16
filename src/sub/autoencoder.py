"""
===========================================================================
                        a u t o e n c o d e r . p y
---------------------------------------------------------------------------
This code is to build and train an autoencoder for feature learning.

Author          : Tomoko Ayakawa
Created on      : 29 March 2019
Last modified on: 16 April 2019
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
# Define the parameters for the autorncoders. When skipped or an invalid 
# value is given, the default values that are defined in conf.py are used.
# -------------------------------------------------------------------------
def get_parameters(data_id):
    from keras.optimizers import Adam, SGD
    
    print("Define the parameters for the autoencoder.\n" \
          "When skipped (push enter) or an invalid value is given, the " \
          "default value will be used.")
    
    # Autoencoder type
    try:
        mode=int(input("[Parameter 1/9: Autoencoder type] "\
             "0:Normal or 1:Stacked (default=0): "))
        if mode not in [0,1]: mode=VAR.ae_mode
    except:
        mode=VAR.ae_mode
    
    # Number of neurons in each layer
    ae_layers=[str(i) for i in VAR.ae_layers[data_id]]
    layers=input("[Parameter 2/9: Number of neurons in each layer] "\
             "Integer separated by comma (default=%s): " % \
             ",".join(ae_layers))
    mod_layers=[]
    try:
        for l in layers.split(","): mod_layers.append(int(l))
    except:
        mod_layers=VAR.ae_layers[data_id]
    
    layers=mod_layers

    # Activation function
    print("[Parameter 3/9: Activation function] ", end="")
    for i in VAR.act_list.keys(): 
        print("%d:%s " % (i, VAR.act_list[i]), end="")  
    act=input("(default=%s): " % VAR.act_list[VAR.ae_act])
    try:
        act=VAR.act_list[int(act)]
    except:
        act=VAR.act_list[VAR.ae_act]

    # Optimiser
    try:
        opt=int(input("[Parameter 4/9: Optimiser function] "\
             "0:adam or 1:sdg (default=%s): " % VAR.ae_opt))
        if opt not in [0,1]: opt=VAR.ae_opt
    except:
       opt=VAR.ae_opt
    if opt==0: #adam
        try:
            lr=float(input(" - Learning rate (default=%f): " % VAR.ae_lr))
        except:
            lr=VAR.ae_lr
        opt=Adam(lr=lr)
    else: # sdg
        try:
            lr=float(input(" - Learning rate (default=%f): " % VAR.ae_lr))
        except:
            lr=VAR.ae_lr
        try:
            momentum=float(input(" - Momentum (default=%f): " % \
                                 VAR.ae_momentum))
        except:
            momentum=VAR.ae_momentum
        opt=SGD(lr=lr, momentum=momentum)
   
    # Loss function
    print("[Parameter 5/9: Loss function] ", end="")
    for i in VAR.loss_list.keys(): 
        print("%d:%s " % (i, VAR.loss_list[i]), end="")
    loss=input("(default=%s): " % VAR.loss_list[VAR.ae_loss])
    try:
        loss=VAR.loss_list[int(loss)]
    except:
        loss=VAR.loss_list[VAR.ae_loss]
    
    # Dropout rate
    try:
        dropout=float(input("[Parameter 6/9: Dropout rate] 0<=rate<1 " \
                              "(default=%f): " % VAR.ae_dropout))
        if (dropout<0) or (dropout>=1): dropout=VAR.ae_dropout
    except:
            dropout=VAR.ae_dropout
       
    # Trainig epochs
    try:
        epochs=int(input("[Parameter 7/9: Training epochs] "\
                         "(default=%d): " % VAR.ae_epoch))
        if epochs<0 : epochs=VAR.ae_epoch
    except:
        epochs=VAR.ae_epoch
    
    
    
    # Verbose
    try:
        verbose=int(input("[Parameter 8/9: Verbose] 0:None or " \
             "1:Display (default=%d): " % VAR.ae_verbose))
        if verbose not in [0,1]: verbose=VAR.ae_verbose
    except:
        verbose=VAR.ae_verbose
    
    # summary_display
    try:
        summary_display=int(input("[Parameter 9/9: Summary display] " \
             "0:False or 1:True (default=%d): " \
             % VAR.ae_summary_display))
        if summary_display not in [0,1]: 
            summary_display=VAR.ae_summary_display
    except:
        summary_display=VAR.ae_summary_display
    
    mode_list={0:"Normal", 1:"Stacked"}
    vb_list={0:"False", 1:"True"}
    print("Parameters for the autoencoder are")
    print(" 1. Mode: %d(%s)" % (mode, mode_list[mode]))
    print(" 2. Layers:", layers)
    print(" 3. Activation function:", act)
    print(" 4. Optimiser:", opt)
    print(" 5. Loss function:", loss)
    print(" 6. Dropout: %f" % dropout)
    print(" 7. Epochs: %f" % epochs)
    print(" 8. Verbose: %d(%s)" % (verbose, vb_list[verbose]))
    print(" 9. Summary_display: %d(%s)" % (summary_display, \
          vb_list[summary_display]))
    
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
                epochs=VAR.ae_epoch, verbose=1, summary_display=False): 
    
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
        
        #build an encoder
        encoder=input_holder
        for i in range (0, n-1):            
            encoder=Dense(layers[i+1], input_dim=layers[i], \
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
        
        if summary_display: print(autoencoder.summary())
        
        return encoder, loss_hiss
    
    # --------------------------------------------------------------------
    # normal autoencoder
    # --------------------------------------------------------------------
    elif mode==1:
        tmp_holder=input_holder
        x=X
        stk_encoders, loss_hist=[],[]
    
        print("layers:", layers)
        #train encoder layers
        for i in range(len(layers)-1):
            print("Training Layer %d/%d ..." % (i+1, len(layers)))
            
            encoder=Dense(layers[i+1], input_dim=layers[i], \
                           activation=act)(tmp_holder)
            if dropout!=0: encoder=Dropout(dropout)(encoder)
                
            decoder=Dense(layers[i], input_dim=layers[i+1], \
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
            tmp_holder=Input(shape=(layers[i+1],))
        
            # store the trainined layer in a list
            stk_encoders.append(encoder)
            
        # connect traind encoder layers as 'encode'
        encoder=Sequential()
        for e in stk_encoders:
            #print(e.summary())
            encoder.add(e)
            print(encoder.summary())
        
        if summary_display: print(encoder.summary())
            
        return encoder, loss_hist
    
# -------------------------------------------------------------------------
# Allow the programme to be ran from the command line.
# -------------------------------------------------------------------------
def plot_ae_loss_history(histories, mode, pic_file):
    ans=input("Save training loss history as a picture? (y/n): ")
    if (ans=="y") or (ans=="Y"): save=True
    else: save=False 

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
    
    layers, mode, act, opt, loss, dropout, epochs, verbose, \
           summary_display=get_parameters(0)
    
    ans=input("Continue? (y/n): ")
    
    if (ans=="y") or (ans=="Y"):
        encode, histories=autoencoder(X, layers=layers, mode=mode,\
                act=act, opt=opt, loss=loss, dropout=dropout, \
                epochs=epochs, verbose=verbose, \
                summary_display=summary_display)
    
    plot_ae_loss_history(histories, mode, "test")
