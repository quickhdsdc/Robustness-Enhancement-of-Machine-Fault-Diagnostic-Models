#=============================================================================#
############################# import functions ################################
#=============================================================================#
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import random
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from scipy.signal import hilbert
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from FeatureExtraction import iF, iFEn_2D, waveletpower

#=============================================================================#
########################### functions #########################################
#=============================================================================#
def input4cnn(data,label,v,ds):         #ds for if downsampling should be performed
    if ds==True:
        n=int(len(data[0,:])/10)
    else:
        n=len(data[0,:])
        
    a_down = np.empty((0,n))  
    for sample in data:
        a_sample=signal.resample(sample,n)
#        [b,a]=signal.butter(2,[2/(fs/2),100/(fs/2)],'bandpass')
#        a_sample=signal.filtfilt(b, a, a_sample)               
        a_sample=preprocessing.scale(a_sample)
        a_sample=np.reshape(a_sample,(1,len(a_sample)))
        a_down=np.append(a_down,a_sample,axis=0)
        
    data_list = a_down  
    rawdata_list = data
    v_list = v   
    x_train, x_test, y_train, y_test,xraw_train, xraw_test,v_train, v_test = train_test_split(data_list,label,rawdata_list, v_list, test_size=0.4)
    x_test, x_val, y_test, y_val,xraw_test,xraw_val,v_test,v_val = train_test_split(x_test,y_test,xraw_test,v_test,test_size=0.5)

    x_train=np.reshape(x_train,(len(x_train),n,1))
    x_test=np.reshape(x_test,(len(x_test),n,1))
    x_val=np.reshape(x_val,(len(x_val),n,1))    
    return x_train, x_test, x_val, y_train, y_test, y_val, xraw_train, xraw_test, xraw_val, v_train, v_test, v_val

def randomize(x,y,xraw,v):
    # Generate the permutation index array.
    permutation = np.random.permutation(x.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    shuffled_x = x[permutation]
    shuffled_y = y[permutation]
    shuffled_xraw = xraw[permutation]
    shuffled_v = v[permutation]    
    return shuffled_x, shuffled_y,shuffled_xraw,shuffled_v
#=============================================================================#
########################### read measurement data signal  #####################
#=============================================================================#
#FolderPath1 = ['Bosch','ESZUEG','LEILA','Entgleis']
FolderPath1 = ['ESZUEG']
FolderPath2 = ['v15v25','v25v35','v35v45','v45v55','v55v65','v65v75','v75v85','v85v95','v95v105']
FolderPath3 = ['Bad','Good']
FolderPath = os.path.abspath(os.path.join(os.getcwd(), "../data/"))
DataPath = os.path.abspath(os.path.join(os.getcwd(), "./axlebox/"))
fs=5000

for i in range(len(FolderPath1)):
    for j in range(len(FolderPath2)):
        for k in range(len(FolderPath3)):
            varname_a = 'a_'+ FolderPath1[i] + '_' +  FolderPath3[k] + '_' + FolderPath2[j] + '.npy'
            varname_v = 'v_'+ FolderPath1[i] + '_' +  FolderPath3[k] + '_' + FolderPath2[j] + '.npy'
            DataPath_a = FolderPath + '/' + varname_a
            DataPath_v = FolderPath + '/' + varname_v
            varname_a1 = 'a_'+ FolderPath1[i] + '_' +  FolderPath3[k] + '_' + FolderPath2[j] 
            varname_v1 = 'v_'+ FolderPath1[i] + '_' +  FolderPath3[k] + '_' + FolderPath2[j]                    
            globals()[varname_a1] = np.load(DataPath_a)
            globals()[varname_v1] = np.load(DataPath_v)
            n = len(globals()[varname_v1])
            
            varname_x_train = 'x_train_' + FolderPath1[i] + '_' +  FolderPath3[k] + '_' + FolderPath2[j] 
            varname_x_val= 'x_val_' + FolderPath1[i] + '_' +  FolderPath3[k] + '_' + FolderPath2[j] 
            varname_x_test= 'x_test_' + FolderPath1[i] + '_' +  FolderPath3[k] + '_' + FolderPath2[j] 
            varname_y_train = 'y_train_' + FolderPath1[i] + '_' +  FolderPath3[k] + '_' + FolderPath2[j] 
            varname_y_val= 'y_val_' + FolderPath1[i] + '_' +  FolderPath3[k] + '_' + FolderPath2[j] 
            varname_y_test= 'y_test_' + FolderPath1[i] + '_' +  FolderPath3[k] + '_' + FolderPath2[j]                 
            varname_xraw_train = 'xraw_train_' + FolderPath1[i] + '_' +  FolderPath3[k] + '_' + FolderPath2[j] 
            varname_xraw_val= 'xraw_val_' + FolderPath1[i] + '_' +  FolderPath3[k] + '_' + FolderPath2[j] 
            varname_xraw_test= 'xraw_test_' + FolderPath1[i] + '_' +  FolderPath3[k] + '_' + FolderPath2[j]                 
            varname_v_train = 'v_train_' + FolderPath1[i] + '_' +  FolderPath3[k] + '_' + FolderPath2[j] 
            varname_v_val= 'v_val_' + FolderPath1[i] + '_' +  FolderPath3[k] + '_' + FolderPath2[j] 
            varname_v_test= 'v_test_' + FolderPath1[i] + '_' +  FolderPath3[k] + '_' + FolderPath2[j] 
            
            if FolderPath3[k]=='Bad': # label is wrong should be inversed. Instead, the test function is revised.
                label=np.hstack((np.reshape(np.ones(n),(n,1)),np.reshape(np.zeros(n),(n,1))))
            else:
                label=np.hstack((np.reshape(np.zeros(n),(n,1)),np.reshape(np.ones(n),(n,1))))
                
            globals()[varname_x_train], globals()[varname_x_test], globals()[varname_x_val], globals()[varname_y_train], globals()[varname_y_test],globals()[varname_y_val], globals()[varname_xraw_train], globals()[varname_xraw_test], globals()[varname_xraw_val], globals()[varname_v_train], globals()[varname_v_test], globals()[varname_v_val] = input4cnn(globals()[varname_a1],label,globals()[varname_v1],False)
#            np.save(DataPath+'/'+varname_x_train, globals()[varname_x_train])
#            np.save(DataPath+'/'+varname_x_test, globals()[varname_x_test])
#            np.save(DataPath+'/'+varname_x_val, globals()[varname_x_val])
#            np.save(DataPath+'/'+varname_y_train, globals()[varname_y_train])
#            np.save(DataPath+'/'+varname_y_test, globals()[varname_y_test])
#            np.save(DataPath+'/'+varname_y_val, globals()[varname_y_val])
#            np.save(DataPath+'/'+varname_xraw_train, globals()[varname_xraw_train])
#            np.save(DataPath+'/'+varname_xraw_test, globals()[varname_xraw_test])
#            np.save(DataPath+'/'+varname_xraw_val, globals()[varname_xraw_val])
#            np.save(DataPath+'/'+varname_v_train, globals()[varname_v_train])
#            np.save(DataPath+'/'+varname_v_test, globals()[varname_v_test])
#            np.save(DataPath+'/'+varname_v_val, globals()[varname_v_val])

###############################################################################
DataPath = os.path.abspath(os.path.join(os.getcwd(), "./axlebox/"))
FolderPath2 = ['v15v25','v25v35','v35v45','v45v55','v55v65','v65v75','v75v85','v85v95','v95v105']
FolderPath4 = ['train','val','test']
FolderPath5 = ['x','y','xraw','v']
for i in range(len(FolderPath4)):
    for j in range(len(FolderPath2)):
        for k in range(len(FolderPath5)):
            varname_Bad = FolderPath5[k] + '_' + FolderPath4[i] + '_ESZUEG_Bad_' + FolderPath2[j]
            varname_Good = FolderPath5[k] + '_' + FolderPath4[i] + '_ESZUEG_Good_' + FolderPath2[j]
            varname =  FolderPath5[k] + '_' + FolderPath4[i] + '_' + FolderPath2[j]
            if FolderPath5[k]=='v':
                globals()[varname]=np.concatenate((globals()[varname_Bad],globals()[varname_Good]))
            else:
                globals()[varname]=np.vstack((globals()[varname_Bad],globals()[varname_Good]))                      
                
        varname_x = 'x_' + FolderPath4[i] + '_' + FolderPath2[j]
        varname_y = 'y_' + FolderPath4[i] + '_' + FolderPath2[j]
        varname_xraw = 'xraw_' + FolderPath4[i] + '_' + FolderPath2[j]
        varname_v = 'v_' + FolderPath4[i] + '_' + FolderPath2[j]
        globals()[varname_x], globals()[varname_y], globals()[varname_xraw], globals()[varname_v]=randomize(globals()[varname_x], globals()[varname_y], globals()[varname_xraw], globals()[varname_v])
#        np.save(DataPath+'/'+varname_x, globals()[varname_x])
#        np.save(DataPath+'/'+varname_y, globals()[varname_y])
#        np.save(DataPath+'/'+varname_xraw, globals()[varname_xraw])
#        np.save(DataPath+'/'+varname_v, globals()[varname_v])

###############################################################################
# choose train, val and test data
DataPath = os.path.abspath(os.path.join(os.getcwd(), "./ESZUEG/train_v35v75/"))

x_train=np.vstack((x_train_v35v45,x_train_v45v55,x_train_v55v65,x_train_v65v75))
v_train=np.concatenate((v_train_v35v45,v_train_v45v55,v_train_v55v65,v_train_v65v75))
y_train=np.vstack((y_train_v35v45,y_train_v45v55,y_train_v55v65,y_train_v65v75))
xraw_train=np.vstack((xraw_train_v35v45,xraw_train_v45v55,xraw_train_v55v65,xraw_train_v65v75))
x_train, v_train, y_train, xraw_train = randomize(x_train, v_train, y_train, xraw_train)
np.save(DataPath+'/x_train',x_train)
np.save(DataPath+'/v_train',v_train)
np.save(DataPath+'/y_train',y_train)
np.save(DataPath+'/xraw_train',xraw_train)

x_val=np.vstack((x_val_v35v45,x_val_v45v55,x_val_v55v65,x_val_v65v75))
v_val=np.concatenate((v_val_v35v45,v_val_v45v55,v_val_v55v65,v_val_v65v75))
y_val=np.vstack((y_val_v35v45,y_val_v45v55,y_val_v55v65,y_val_v65v75))
xraw_val=np.vstack((xraw_val_v35v45,xraw_val_v45v55,xraw_val_v55v65,xraw_val_v65v75))
x_val, v_val, y_val, xraw_val = randomize(x_val, v_val, y_val, xraw_val)
np.save(DataPath+'/x_val',x_val)
np.save(DataPath+'/v_val',v_val)
np.save(DataPath+'/y_val',y_val)
np.save(DataPath+'/xraw_val',xraw_val)

x_test=np.vstack((x_test_v35v45,x_test_v45v55,x_test_v55v65,x_test_v65v75))
v_test=np.concatenate((v_test_v35v45,v_test_v45v55,v_test_v55v65,v_test_v65v75))
y_test=np.vstack((y_test_v35v45,y_test_v45v55,y_test_v55v65,y_test_v65v75))
xraw_test=np.vstack((xraw_test_v35v45,xraw_test_v45v55,xraw_test_v55v65,xraw_test_v65v75))
x_test, v_test, y_test, xraw_test = randomize(x_test, v_test, y_test, xraw_test)
np.save(DataPath+'/x_test',x_test)
np.save(DataPath+'/v_test',v_test)
np.save(DataPath+'/y_test',y_test)
np.save(DataPath+'/xraw_test',xraw_test)
###############################################################################
#DataPath = os.path.abspath(os.path.join(os.getcwd(), "./ESZUEG/test_v85v95/"))
#x_test=x_test_v85v95
#v_test=v_test_v85v95
#y_test=y_test_v85v95
#xraw_test=xraw_test_v85v95
#x_test, v_test, y_test, xraw_test = randomize(x_test, v_test, y_test, xraw_test)
#np.save(DataPath+'/x_test',x_test)
#np.save(DataPath+'/v_test',v_test)
#np.save(DataPath+'/y_test',y_test)
#np.save(DataPath+'/xraw_test',xraw_test)

#=============================================================================#
########################### plus synthetic data ###############################
#=============================================================================#
FolderPath_syn = os.path.abspath(os.path.join(os.getcwd(), "../data"))
FolderPath1 = ['v110','v90','v70','v50','v20']
FolderPath2 = ['normal','WF']
for i in range(len(FolderPath1)):
    for j in range(len(FolderPath2)):
        FileName = 'syn_'+ FolderPath2[j] + '_' + FolderPath1[i] + '.npy'
        FilePath = os.path.join(FolderPath_syn, FileName)
        VarName = 'syn_'+ FolderPath2[j] + '_' + FolderPath1[i]
        globals()[VarName] = np.load(FilePath)
        
        
