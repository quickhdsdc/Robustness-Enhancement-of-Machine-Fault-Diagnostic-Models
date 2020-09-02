#=============================================================================#
############################# import functions ################################
#=============================================================================#
import numpy as np
import os
import pandas as pd
#=============================================================================#
############################ functions ########################################
#=============================================================================#
def segment_signal(data,window_size = 10000):
    a = np.empty((0,window_size))
    v = np.empty((0))
    for (start, end) in windows(data['az'], window_size,0.5):
        segment = data['az'][start:end]
        speed = np.mean(data['v'][start:end])
        if(len(data['az'][start:end]) == window_size):
            a = np.vstack([a,segment])
            v = np.append(v,speed)
    return a, v                    

def windows(data, size, overlap):
    start = 0
    while start < data.count():
        yield int(start), int(start + size)
        start += (size*overlap)
        
#=============================================================================#
############################ read data: Bosch and ESZUEG ######################
#=============================================================================# 
        
fs = 5000
FolderPath = os.path.abspath(os.path.join(os.getcwd(), "../Continous Raw Data/"))
FolderPath_save = os.path.abspath(os.path.join(os.getcwd(), "../data/"))
FolderPath1 = ['Bosch','ESZUEG']
FolderPath2 = ['Axlebox over wheelflat']
FolderPath3 = ['v15v25','v25v35','v35v45','v45v55','v55v65','v65v75','v75v85','v85v95','v95v105']
FolderPath4 = ['Bad','Good']

for i in range(len(FolderPath1)):
    for j in range(len(FolderPath2)):
        for k in range(len(FolderPath3)):
            for l in range(len(FolderPath4)):
#                if FolderPath1[i]=='Bosch' and FolderPath4[l]=='Bad':
#                     DirPath = FolderPath + '/' +  FolderPath1[i] + '/' + FolderPath2[j] + '/Velocity sorted rides/'+ FolderPath3[k] + '/' + FolderPath4[l] + '/10mm'                                         
#                else:
                DirPath = FolderPath + '/' +  FolderPath1[i] + '/' + FolderPath2[j] + '/Velocity sorted rides/'+ FolderPath3[k] + '/' + FolderPath4[l]                      
                varname_a = 'a_'+ FolderPath1[i] + '_' +  FolderPath4[l] + '_' + FolderPath3[k]
                varname_v = 'v_'+ FolderPath1[i] + '_' +  FolderPath4[l] + '_' + FolderPath3[k]
                globals()[varname_a] = np.empty((0,fs*2))
                globals()[varname_v] = np.empty((0))
                for file in os.listdir(DirPath):
                    if file.endswith('.csv'):
                        VarName, ext = os.path.splitext(file)
                        FilePath = os.path.join(DirPath, file)  
                        df = pd.read_csv(FilePath)
                        a,v=segment_signal(df,window_size = fs*2)
                        globals()[varname_a] = np.append(globals()[varname_a],a,axis=0)
                        globals()[varname_v] = np.append(globals()[varname_v],v)
                DataPath_a = FolderPath_save + '/' + varname_a
                DataPath_v = FolderPath_save + '/' + varname_v
                np.save(DataPath_a, globals()[varname_a])
                np.save(DataPath_v, globals()[varname_v])

#=============================================================================#
########################## read data: Entgleis and LEILA ######################
#=============================================================================#                             
fs = 500
FolderPath = os.path.abspath(os.path.join(os.getcwd(), "../Continous Raw Data/"))
FolderPath_save = os.path.abspath(os.path.join(os.getcwd(), "../data/"))
FolderPath1 = ['LEILA']
FolderPath2 = ['Axlebox over wheelflat']
#FolderPath3 = ['v15v25','v25v35','v35v45','v45v55','v55v65','v65v75','v75v85','v85v95','v95v105']
FolderPath3 = ['v15v25','v25v35','v35v45','v45v55','v55v65','v65v75','v75v85']
FolderPath4 = ['Bad','Good']

for i in range(len(FolderPath1)):
    for j in range(len(FolderPath2)):
        for k in range(len(FolderPath3)):
            for l in range(len(FolderPath4)):
                DirPath = FolderPath + '/' +  FolderPath1[i] + '/' + FolderPath2[j] + '/Velocity sorted rides/'+ FolderPath3[k] + '/' + FolderPath4[l]                      
                varname_a = 'a_'+ FolderPath1[i] + '_' +  FolderPath4[l] + '_' + FolderPath3[k]
                varname_v = 'v_'+ FolderPath1[i] + '_' +  FolderPath4[l] + '_' + FolderPath3[k]
                globals()[varname_a] = np.empty((0,fs*2))
                globals()[varname_v] = np.empty((0))
                for file in os.listdir(DirPath):
                    if file.endswith('.csv'):
                        VarName, ext = os.path.splitext(file)
                        FilePath = os.path.join(DirPath, file)  
                        df = pd.read_csv(FilePath)
                        a,v=segment_signal(df,window_size = fs*2)
                        globals()[varname_a] = np.append(globals()[varname_a],a,axis=0)
                        globals()[varname_v] = np.append(globals()[varname_v],v)
                DataPath_a = FolderPath_save + '/' + varname_a
                DataPath_v = FolderPath_save + '/' + varname_v
                np.save(DataPath_a, globals()[varname_a])
                np.save(DataPath_v, globals()[varname_v])                        
                    
                        
#=============================================================================#
########################## read data: simulation ##############################
#=============================================================================#            
FolderPath = os.path.abspath(os.path.join(os.getcwd(), "../data/"))
sim_normal=pd.read_csv(FolderPath+'/simdata_normal.txt',sep='\s+',engine='python')
sim_WF=pd.read_csv(FolderPath+'/simdata_WF.txt',sep='\s+',engine='python')                

def segment_signal(data,col,window_size = 10000):
    a = np.empty((0,window_size))
    for (start, end) in windows(data[col], window_size,0.5):
        segment = data[col][start:end]
        if(len(data[col][start:end]) == window_size):
            a = np.vstack([a,segment])
    return a   

def windows(data, size, overlap):
    start = 0
    while start < data.count():
        yield int(start), int(start + size)
        start += (size*overlap)


def SNR_Noise(signal,SNR_db):
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

        
FolderPath1 = ['v110','v90','v70','v50','v20']
FolderPath2 = ['normal','WF']
for i in range(len(FolderPath1)):
    for j in range(len(FolderPath2)):
        a_varname = 'sim_'+ FolderPath2[j] + '_' +  FolderPath1[i]
        df_varname = 'sim_'+ FolderPath2[j]
        globals()[a_varname]=segment_signal(globals()[df_varname],FolderPath1[i],10000)
        for n in range(100):
            globals()[a_varname]=np.append(globals()[a_varname],globals()[a_varname].copy())
            

sim_WF_v50_noise=SNR_Noise(sim_WF_v50,0)
a_input_gaussian=a_ref+noise