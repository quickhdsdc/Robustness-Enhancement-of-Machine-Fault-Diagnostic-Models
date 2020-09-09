#=============================================================================#
############################# import functions ################################
#=============================================================================#
import numpy as np
import os
import pandas as pd
from sklearn import preprocessing
#import matplotlib.pyplot as plt
from scipy import signal  
from ultis import *          
#=============================================================================#
########################## read data: simulation ##############################
#=============================================================================#            
FolderPath = os.path.abspath(os.path.join(os.getcwd(), "./input_data/"))
sim_normal=pd.read_csv(FolderPath+'/simdata_normal.txt',sep='\s+',engine='python')
sim_WF=pd.read_csv(FolderPath+'/simdata_WF.txt',sep='\s+',engine='python')                
      
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
#################### produce data: normal synthetic data ######################
#=============================================================================#                
for file in os.listdir(FolderPath):
    if file.endswith('.npy'):
        VarName, ext = os.path.splitext(file)
        FilePath = os.path.join(FolderPath, file)
        globals()[VarName] = np.load(FilePath)

x_syn_peak, y_syn_peak=augment_train_set(xraw_test_peak[0:49,:], y_test_peak[0:49,:],100)
x_syn_per, y_syn_per=augment_train_set(xraw_test_per[0:49,:], y_test_per[0:49,:],100)
x_syn_osc, y_syn_osc=augment_train_set(xraw_test_osc[0:49,:], y_test_osc[0:49,:],100)

FolderPath_output = os.path.abspath(os.path.join(os.getcwd(), "./output_data/"))
np.save(FolderPath_output+"/x_syn_peak.npy",x_syn_peak)
np.save(FolderPath_output+"/x_syn_per.npy",x_syn_per)
np.save(FolderPath_output+"/x_syn_osc.npy",x_syn_osc)

x_syn_peak=signal.resample(x_syn_peak,10000,axis=1)
x_syn_peak = preprocessing.scale(x_syn_peak)
n_syn_norm= len(x_syn_peak)

x_syn_per=signal.resample(x_syn_per,10000,axis=1)
x_syn_per = preprocessing.scale(x_syn_per)

x_syn_osc=signal.resample(x_syn_osc,10000,axis=1)
x_syn_osc = preprocessing.scale(x_syn_osc)
#=============================================================================#
########### produce data: reality-augmented simulation faulty data ############
#=============================================================================# 
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
        FilePath=os.path.abspath(os.path.join(os.getcwd(), "./output_data/"+syn_varname))
        np.save(FilePath,globals()[syn_varname])
        
        
