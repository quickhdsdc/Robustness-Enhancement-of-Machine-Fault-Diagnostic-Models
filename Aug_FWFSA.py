#=============================================================================#
############################# import functions ################################
#=============================================================================#
import numpy as np
import os
import pandas as pd
from scipy import signal  
from scipy.signal import resample
from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler()
from Utils.utils_aug_ft import augment_train_set_ft, segment_signal
from Utils.FeatureExtraction import ExtraFeatures
#=============================================================================#
########################## read data: simulation ##############################
#=============================================================================#     
## 20 mm wheel flat simulation data       
FolderPath = os.path.abspath(os.path.join(os.getcwd(), "./Data/Augmentation/input_data/"))
sim_normal=pd.read_csv(FolderPath+'/sim_norm_20mm.txt',engine='python') # can also be 30mm or 50mm WF
sim_WF=pd.read_csv(FolderPath+'/sim_WF_20mm.txt',engine='python')                

# time shifting for more data samples     
FolderPath1 = ['v70'] # choose the speed from dataframe
FolderPath2 = ['normal','WF']
FolderPath3 = ['Good','Bad']
for i in range(len(FolderPath1)):
    for j in range(len(FolderPath2)):
        a_varname = 'sim_'+ FolderPath2[j] + '_' +  FolderPath1[i]
        df_varname = 'sim_'+ FolderPath2[j]
        globals()[a_varname]=segment_signal(globals()[df_varname],FolderPath1[i],10240)
# flipping for doubling data samples              
for i in range(len(FolderPath1)):
    for j in range(len(FolderPath2)):
        a_varname = 'sim_'+ FolderPath2[j] + '_' +  FolderPath1[i]      
        n_sample = len(globals()[a_varname])
        temp=np.copy(globals()[a_varname])
        globals()[a_varname]=np.append(globals()[a_varname],temp*-1,axis=0)
        globals()[a_varname]=scaler.fit_transform(globals()[a_varname].T).T
        globals()[a_varname] = globals()[a_varname] - np.mean(globals()[a_varname],axis=1, keepdims=True)

#=============================================================================#
#################### produce data: normal synthetic data ######################
#=============================================================================#                
for file in os.listdir(FolderPath):
    if file.endswith('.npy'):
        VarName, ext = os.path.splitext(file)
        FilePath = os.path.join(FolderPath, file)
        globals()[VarName] = np.load(FilePath)
###
# fwfsa 
def FWDBA_aug(x,fs,N):  
    x = resample(x,int(2.048*fs),axis=1)
    indices = list(range(len(x)))  
    np.random.shuffle(indices)
    x = x[indices[:49]]
    x_ft = []    
    for n in range(len(x)):                  
        raw = x[n]                                           
        features = ExtraFeatures(raw,fs)
        x_ft.append(features)
    x_ft = np.asarray(x_ft)
    x_syn = augment_train_set_ft(x, x_ft, N)
    return x_syn


n_syn=500 # synthetic time series to be produced
x_syn_peak = FWDBA_aug(xraw_test_peak, 5000, n_syn//5)
x_syn_per = FWDBA_aug(xraw_test_per, 5000, n_syn//5)
x_syn_osc = FWDBA_aug(xraw_test_osc, 5000, n_syn//5)

##
x_syn_peak=signal.resample(x_syn_peak,10240,axis=1)
x_syn_peak = scaler.fit_transform(x_syn_peak.T).T
x_syn_peak = x_syn_peak - np.mean(x_syn_peak,axis=1, keepdims=True)
n_syn_norm= len(x_syn_peak)

x_syn_per=signal.resample(x_syn_per,10240,axis=1)
x_syn_per = scaler.fit_transform(x_syn_per.T).T
x_syn_per = x_syn_per - np.mean(x_syn_per,axis=1, keepdims=True)

x_syn_osc=signal.resample(x_syn_osc,10240,axis=1)
x_syn_osc = scaler.fit_transform(x_syn_osc.T).T
x_syn_osc = x_syn_osc - np.mean(x_syn_osc,axis=1, keepdims=True)

#=============================================================================#
########### produce data: reality-augmented simulation faulty data ############
#=============================================================================# 
n_syn_norm= len(x_syn_peak)
n_syn=500
Syn_norm = ['peak','per','osc']
FolderPath1 = ['v70']
FolderPath2 = ['normal','WF']
FolderPath3 = ['Good','Bad']
for i in range(len(FolderPath1)):
    for j in range(len(FolderPath2)):
        a_varname = 'sim_'+ FolderPath2[j] + '_' +  FolderPath1[i]
        syn_varname = 'a_syn_'+ FolderPath3[j] + '_' +  FolderPath1[i]
        globals()[syn_varname] = np.zeros((n_syn,10240))
        n_sim = len(globals()[a_varname])
        indics=np.random.randint(0,n_sim,n_syn)
        for n in range(n_syn):
            temp_sim=globals()[a_varname][indics[n]]
            # temp_sim=SNR_Noise_1D(temp_sim,20)
            if FolderPath2[j]=='WF':
                index=int(np.random.randint(0,n_syn_norm,1))
                syn_type=int(np.random.randint(0,2,1))
                temp_syn_varname='x_syn_'+Syn_norm[syn_type]
                temp_syn=globals()[temp_syn_varname][index]
                globals()[syn_varname][n]=temp_sim+temp_syn*(0.5+i*0.5)
            else:
                syn_type=int(np.random.randint(0,2,1))
                index=int(np.random.randint(0,n_syn_norm,1))
                temp_syn_varname='x_syn_'+Syn_norm[syn_type]
                temp_syn=globals()[temp_syn_varname][index]                
                globals()[syn_varname][n]=temp_sim
    