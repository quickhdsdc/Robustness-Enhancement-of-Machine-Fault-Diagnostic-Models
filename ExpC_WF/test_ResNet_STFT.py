#=============================================================================#
############################# import functions ################################
#=============================================================================#
from keras import backend as K
import os
import numpy as np
#import matplotlib.pyplot as plt
from FeatureExtraction import stft_powerspectrum
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier 
from models import *
from utils import *

#=============================================================================#
###################### read data and add synthetic data #######################
#=============================================================================#
# read measurement data
FolderPath_train = os.path.abspath(os.path.join(os.getcwd(), "./data/train_ESZUEG_500"))
for file in os.listdir(FolderPath_train):
    if file.endswith('.npy'):
        VarName, ext = os.path.splitext(file)
        FilePath = os.path.join(FolderPath_train, file)
        locals()[VarName] = np.load(FilePath)

#=============================================================================#
###################### add synthetic data #####################################
#=============================================================================#
#n_syn=500
#x_train_syn=np.zeros((n_syn*6,1000,1))  
#y_train_syn=np.zeros((n_syn*6,2))
#count=0
#FolderPath_syn = os.path.abspath(os.path.join(os.getcwd(), "../Augmentation/output_data"))
#FolderPath1 = ['v50','v20','v70']
#FolderPath2 = ['WF','normal']
#for i in range(len(FolderPath1)):
#    for j in range(len(FolderPath2)):
#        FileName = 'syn_'+ FolderPath2[j] + '_' + FolderPath1[i] + '.npy'
#        FilePath = os.path.join(FolderPath_syn, FileName)
#        VarName = 'syn_'+ FolderPath2[j] + '_' + FolderPath1[i]
#        globals()[VarName] = np.load(FilePath)                
#        count=count+1
#        x_train_syn[int((count-1)*n_syn):int((count)*n_syn)]=np.reshape(globals()[VarName][:n_syn],(n_syn,1000,1))
#        if FolderPath2[j]=='normal':
#            y_train_syn[int((count-1)*n_syn):int((count)*n_syn),1]=1
#        else:
#            y_train_syn[int((count-1)*n_syn):int((count)*n_syn),0]=1
#                  
#x_train=np.append(x_train,x_train_syn,axis=0)        
#y_train=np.append(y_train,y_train_syn,axis=0)   


#=============================================================================#
#################################### STFT #####################################
#=============================================================================#
fs=500    
#fs=5000    
x_train=stft_powerspectrum(x_train,fs)
x_val=stft_powerspectrum(x_val,fs)
x_test=stft_powerspectrum(x_test,fs)  

#=============================================================================#
######################### train and test for 10 times #########################
#=============================================================================#
FolderPath = ['train_ESZUEG_500','test_Entgleis','test_LEILA']
#FolderPath = ['train_ESZUEG','test_Bosch']

metrics_train_ESZUEG_500=np.zeros((10,4))
#metrics_train_ESZUEG=np.zeros((10,4))
#metrics_test_Bosch=np.zeros((10,4))
metrics_test_Entgleis=np.zeros((10,4))
metrics_test_LEILA=np.zeros((10,4))

global best_criteria
best_criteria=0

for n in range(0,10):
    print('run: ' + str(n))    
    try:
        del model
        K.clear_session
    except:
        print('model is cleaned')
        pass   
#    model=ResNet_STFT((80,80,1))
    model=ResNet_STFT((33,33,1))                                                   # build model
    model.fit(x_train,y_train,batch_size=32,epochs=20,validation_data=[x_val, y_val])  #fiting with hyperparameters 

    for i in range(3):
#    for i in range(2):        
        FolderPath_test = os.path.abspath(os.path.join(os.getcwd(), ("./data/"+FolderPath[i])))
        for file in os.listdir(FolderPath_test):
            if file.endswith('.npy'):
                VarName, ext = os.path.splitext(file)
                FilePath = os.path.join(FolderPath_test, file)
                locals()[VarName] = np.load(FilePath)   

        if len(x_test[0,:,0])==2*fs:
            x_test=stft_powerspectrum(x_test,fs)                                              # STFT 
        if len(x_train[0,:,0])==2*fs:
            x_train=stft_powerspectrum(x_train,fs)          
        if len(x_val[0,:,0])==2*fs:
            x_val=stft_powerspectrum(x_val,fs)  

        acc,_,_,_,_=test_CNN(model,x_test,y_test)
        Varname_metrics = 'metrics_' + FolderPath[i]
        locals()[Varname_metrics][n,0]=acc
#        locals()[Varname_metrics][n,1]=recall
#        locals()[Varname_metrics][n,2]=precision
#        locals()[Varname_metrics][n,3]=auc        
#    criteria=np.mean(metrics_train_v55v95[n,0])
#    if criteria>best_criteria:
#        best_criteria=criteria
#        model.save('./CNN_AB_v55v95_run'+str(n))
#        np.save('train_acc_best',history.history['acc'])
#        np.save('val_acc_best',history.history['val_acc'])
#        np.save('train_loss_best',history.history['loss'])
#        np.save('val_loss_best',history.history['val_loss'])
        #summarize history for accuracy
    
# read the training data again    
    FolderPath_train = os.path.abspath(os.path.join(os.getcwd(), "./data/train_ESZUEG_500"))
    for file in os.listdir(FolderPath_train):
        if file.endswith('.npy'):
            VarName, ext = os.path.splitext(file)
            FilePath = os.path.join(FolderPath_train, file)
            locals()[VarName] = np.load(FilePath)
        
    n_syn=500
    x_train_syn=np.zeros((n_syn*8,1000,1))  
    y_train_syn=np.zeros((n_syn*8,2))
    count=0
    FolderPath_syn = os.path.abspath(os.path.join(os.getcwd(), "../Augmentation/output_data"))
    FolderPath1 = ['v90','v70','v50','v20']
    FolderPath2 = ['WF','normal']
    for i in range(len(FolderPath1)):
        for j in range(len(FolderPath2)):
            FileName = 'syn_'+ FolderPath2[j] + '_' + FolderPath1[i] + '.npy'
            FilePath = os.path.join(FolderPath_syn, FileName)
            VarName = 'syn_'+ FolderPath2[j] + '_' + FolderPath1[i]
            globals()[VarName] = np.load(FilePath)                
            count=count+1
            x_train_syn[int((count-1)*n_syn):int((count)*n_syn)]=np.reshape(globals()[VarName][:n_syn],(n_syn,1000,1))
            if FolderPath2[j]=='normal':
                y_train_syn[int((count-1)*n_syn):int((count)*n_syn),1]=1
            else:
                y_train_syn[int((count-1)*n_syn):int((count)*n_syn),0]=1
                      
    x_train=np.append(x_train,x_train_syn,axis=0)        
    y_train=np.append(y_train,y_train_syn,axis=0)      
    # STFT
    fs=500    
    x_train=stft_powerspectrum(x_train,fs)
    x_val=stft_powerspectrum(x_val,fs)
    x_test=stft_powerspectrum(x_test,fs)  