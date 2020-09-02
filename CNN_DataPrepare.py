#=============================================================================#
############################# import functions ################################
#=============================================================================#
import numpy as np
import os
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy import signal            
#=============================================================================#
########################## read data: simulation ##############################
#=============================================================================#            
FolderPath = os.path.abspath(os.path.join(os.getcwd(), "../data/"))
sim_normal=pd.read_csv(FolderPath+'/simdata_normal.txt',sep='\s+',engine='python')
sim_WF=pd.read_csv(FolderPath+'/simdata_WF.txt',sep='\s+',engine='python')                

def segment_signal(data,col,window_size = 10000):
    a = np.empty((0,window_size))
    for index_start in range(0,data[col].count()-window_size,window_size//50):
        for (start, end) in windows(data[col], window_size,0.5,index_start):
            segment = data[col][start:end]
            if(len(data[col][start:end]) == window_size):
                a = np.vstack([a,segment])
    return a   

def windows(data, size, overlap,start):
    while start < data.count():
        yield int(start), int(start + size)
        start += (size*overlap)


def SNR_Noise_2D(signal,SNR_db):
    Npts = np.shape(signal)[1]
    num = np.shape(signal)[0]
    signal_noise=np.zeros((num,Npts))
    for i in range(num):
        sig=signal[i]
        noise = np.random.normal(0,1,Npts)
        sig_power = np.mean(sig**2)
        noise_power=np.mean(noise**2)
        K=(sig_power/noise_power)*10**(-SNR_db/10)
        new_noise=np.sqrt(K)*noise
        sig = sig + new_noise
        signal_noise[i]= sig       
    return signal_noise  

def SNR_Noise_1D(signal,SNR_db):
    Npts = np.shape(signal)[0]
    signal_noise=np.zeros((Npts))
    sig=signal
    noise = np.random.normal(0,1,Npts)
    sig_power = np.mean(sig**2)
    noise_power=np.mean(noise**2)
    K=(sig_power/noise_power)*10**(-SNR_db/10)
    new_noise=np.sqrt(K)*noise
    sig = sig + new_noise
    signal_noise= sig       
    return signal_noise 
       
FolderPath1 = ['v110','v90','v70','v50','v20']
FolderPath2 = ['normal','WF']
for i in range(len(FolderPath1)):
    for j in range(len(FolderPath2)):
        a_varname = 'sim_'+ FolderPath2[j] + '_' +  FolderPath1[i]
        df_varname = 'sim_'+ FolderPath2[j]
        globals()[a_varname]=segment_signal(globals()[df_varname],FolderPath1[i],10000)
            
for i in range(len(FolderPath1)):
    for j in range(len(FolderPath2)):
        a_varname = 'sim_'+ FolderPath2[j] + '_' +  FolderPath1[i]      
        n_sample = len(globals()[a_varname])
#        globals()[a_varname]=signal.resample(globals()[a_varname],1000,axis=1)       
        temp=np.copy(globals()[a_varname])
        globals()[a_varname]=np.append(globals()[a_varname],temp*-1,axis=0)
        globals()[a_varname]=preprocessing.scale(globals()[a_varname])
 
#=============================================================================#
############### read data: synthetic faulty data ##############################
#=============================================================================#                
FolderPath_train = os.path.abspath(os.path.join(os.getcwd(), "./data_syn"))
for file in os.listdir(FolderPath_train):
    if file.endswith('.npy'):
        VarName, ext = os.path.splitext(file)
        FilePath = os.path.join(FolderPath_train, file)
        globals()[VarName] = np.load(FilePath)
        globals()[VarName]=signal.resample(globals()[VarName],10000,axis=1)
        globals()[VarName] = preprocessing.scale(globals()[VarName])
        n_syn_norm= len(globals()[VarName])

n_syn=2000 # synthetic time series to be produced
Syn_norm = ['peak','per']
FolderPath1 = ['v110','v90','v70','v50','v20']
FolderPath2 = ['normal','WF']
for i in range(len(FolderPath1)):
    for j in range(len(FolderPath2)):
        a_varname = 'sim_'+ FolderPath2[j] + '_' +  FolderPath1[i]
        syn_varname = 'syn_'+ FolderPath2[j] + '_' +  FolderPath1[i]
        globals()[syn_varname] = np.zeros((n_syn,10000))
        n_sim = len(globals()[a_varname])
        indics=np.random.randint(0,n_sim,n_syn)
        for n in range(n_syn):
            temp_sim=globals()[a_varname][indics[n]]
            temp_sim=SNR_Noise_1D(temp_sim,10)
            if FolderPath2[j]=='WF':
                index=int(np.random.randint(0,n_syn_norm,1))
                temp_syn_varname='x_syn_'+Syn_norm[0]
                temp_syn=globals()[temp_syn_varname][index]
                globals()[syn_varname][n]=temp_sim
            else:
                syn_type=int(np.random.randint(0,2,1))
                index=int(np.random.randint(0,n_syn_norm,1))
                temp_syn_varname='x_syn_'+Syn_norm[syn_type]
                temp_syn=globals()[temp_syn_varname][index]                
                globals()[syn_varname][n]=temp_sim
        FilePath=os.path.abspath(os.path.join(os.getcwd(), "../data/"+syn_varname))
        np.save(FilePath,globals()[syn_varname])
        
        
