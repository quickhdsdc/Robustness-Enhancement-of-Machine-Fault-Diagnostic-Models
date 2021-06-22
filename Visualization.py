from Utils.UTILS import *
from Utils.FeatureExtraction import *
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
#=============================================================================#
############################### domain distance ###############################
#=============================================================================#
folderpath = './Data/MMD_env/'
for file in os.listdir(folderpath):
    if file.endswith('.npy'):
        varname, ext = os.path.splitext(file)
        filepath = os.path.join(folderpath, file)
        globals()[varname] = np.load(filepath)

####### speed variation        
# ESZUEG
cm = 1/2.54  # centimeters in inches
histo={'v25v35':[mmd_ESZUEG_Bad_env_v15v25_all[0], mmd_ESZUEG_Good_env_v15v25_all[0]],'v35v45':[mmd_ESZUEG_Bad_env_v15v25_all[1], mmd_ESZUEG_Good_env_v15v25_all[1]],'v45v55':[mmd_ESZUEG_Bad_env_v15v25_all[2], mmd_ESZUEG_Good_env_v15v25_all[2]],
       'v55v65':[mmd_ESZUEG_Bad_env_v15v25_all[3], mmd_ESZUEG_Good_env_v15v25_all[3]],'v65v75':[mmd_ESZUEG_Bad_env_v15v25_all[4], mmd_ESZUEG_Good_env_v15v25_all[4]],'v75v85':[mmd_ESZUEG_Bad_env_v15v25_all[5], mmd_ESZUEG_Good_env_v15v25_all[5]],
       'v85v95':[mmd_ESZUEG_Bad_env_v15v25_all[6], mmd_ESZUEG_Good_env_v15v25_all[6]],'v95v105':[mmd_ESZUEG_Bad_env_v15v25_all[7], mmd_ESZUEG_Good_env_v15v25_all[7]]}
df=pd.DataFrame.from_dict(histo, orient='index',columns=['M1_Bad','M1_Good'])
fig=df.plot(kind='line',color=['#fb0000','#0000fb'],marker='D',figsize=(8.65*cm*2, 6.35*cm*2))
plt.legend(bbox_to_anchor=(0.5, 1.08), loc='upper center', fontsize='large', ncol=2,handletextpad=0.3,borderpad=0.2)                          
plt.ylabel('MMD', fontsize='large')
plt.xlabel('Speed range (km/h)', fontsize='large')
plt.tight_layout()
plt.grid(True) 
fig.figure.savefig('./Figures/MMD_SpeedVariation_ESZUEG.png',dpi=900) 

# LEILA
histo={'v25v35':[mmd_LEILA_Bad_env_v15v25_all[0], mmd_LEILA_Good_env_v15v25_all[0]],'v35v45':[mmd_LEILA_Bad_env_v15v25_all[1], mmd_LEILA_Good_env_v15v25_all[1]],'v45v55':[mmd_LEILA_Bad_env_v15v25_all[2], mmd_LEILA_Good_env_v15v25_all[2]],
       'v55v65':[mmd_LEILA_Bad_env_v15v25_all[3], mmd_LEILA_Good_env_v15v25_all[3]],'v65v75':[mmd_LEILA_Bad_env_v15v25_all[4], mmd_LEILA_Good_env_v15v25_all[4]],'v75v85':[mmd_LEILA_Bad_env_v15v25_all[5], mmd_LEILA_Good_env_v15v25_all[5]]}
df=pd.DataFrame.from_dict(histo, orient='index',columns=['M3_Bad','M3_Good'])
fig=df.plot(kind='line',color=['#fb0000','#0000fb'],marker='D',figsize=(8.65*cm*2, 6.35*cm*2))
plt.legend(bbox_to_anchor=(0.5, 1.08), loc='upper center', fontsize='large', ncol=2,handletextpad=0.3,borderpad=0.2)                          
plt.ylabel('MMD', fontsize='large')
plt.xlabel('Speed range (km/h)', fontsize='large')
plt.tight_layout()
plt.grid(True) 
fig.figure.savefig('./Figures/MMD_SpeedVariation_LEILA.png',dpi=900) 

####### object variation
# v25v35
histo={'M1-M2':[mmd_Bad_v25v35_env_ESZUEG_all[0], mmd_Good_v25v35_env_ESZUEG_all[0]],'M1-M3':[mmd_Bad_v25v35_env_ESZUEG_all[1], mmd_Good_v25v35_env_ESZUEG_all[1]],'M1-M4':[mmd_Bad_v25v35_env_ESZUEG_all[2]]}
df=pd.DataFrame.from_dict(histo, orient='index',columns=['v25v35_Bad','v25v35_Good'])
fig=df.plot(kind='line',color=['#fb0000','#0000fb'],marker='D',figsize=(8.65*cm*2, 6.35*cm*2))
plt.legend(bbox_to_anchor=(0.5, 1.08), loc='upper center', fontsize='large', ncol=2,handletextpad=0.3,borderpad=0.2)                          
plt.ylabel('MMD', fontsize='large')
plt.xlabel('Dataset pair', fontsize='large')
plt.tight_layout()
plt.grid(True) 
plt.ylim((0,0.03))
fig.figure.savefig('./Figures/MMD_DatasetVariation_v25v35.png',dpi=900) 

####### signal interference variation
histo={'T1_WFbad':[mmd_ESZUEG_trackirr[0]],'T2_osc':[mmd_ESZUEG_trackirr[1]],'T3_peak':[mmd_ESZUEG_trackirr[2]],'T4_per':[mmd_ESZUEG_trackirr[3]]}
df=pd.DataFrame.from_dict(histo, orient='index',columns=['M1_v35v75_Bad'])
fig=df.plot(kind='line',color=['#fb0000'],marker='D',figsize=(8.65*cm*2, 6.35*cm*2))
plt.legend(bbox_to_anchor=(0.5, 1.08), loc='upper center', fontsize='large', ncol=2,handletextpad=0.3,borderpad=0.2)                          
plt.ylabel('MMD', fontsize='large')
plt.xlabel('Interference type', fontsize='large')
plt.tight_layout()
plt.grid(True) 
fig.figure.savefig('./Figures/MMD_InterfVariation_v35v75.png',dpi=900) 

####### intra-class vs inter-class
mmd_ESZUEG_Good_Bad_env = np.mean(mmd_ESZUEG_Good_Bad_env_all)
histo={'inter-class':[mmd_ESZUEG_Good_Bad_env],'speed variation':[mmd_ESZUEG_Bad_env_v15v25_all[6]],'object variation':[mmd_Bad_v25v35_env_ESZUEG_all[1]],'interference variation':[mmd_ESZUEG_trackirr[3]]}
df=pd.DataFrame.from_dict(histo, orient='index',columns=['M1'])
fig=df.plot(kind='line',color=['#fb0000'],marker='D',figsize=(8.65*cm*2, 6.35*cm*2))
plt.legend(bbox_to_anchor=(0.5, 1.08), loc='upper center', fontsize='large', ncol=6,handletextpad=0.3,borderpad=0.2)                          
plt.ylabel('MMD', fontsize='large')
plt.tight_layout()
plt.grid(True) 
fig.figure.savefig('./Figures/MMD_inter_intra_class.png',dpi=900) 

#=============================================================================#
###################### Robustness speed variation #############################
#=============================================================================#
histo={'55-105':[99.85, 99.83, 99.28, 99.41],'45-55':[98.42, 96.60, 93.59, 95.92],'35-45':[96.88, 96.79, 94.21, 96.35],'25-35':[80.95, 82.47, 78.57, 92.57],'15-25':[58.00, 56.10, 59.93, 88.34]}
df=pd.DataFrame.from_dict(histo, orient='index',columns=['ResNet','CWT+ResNet','HT+ResNet','GBDT'])
fig=df.plot(kind='line',color=['#E41352','#736D6D','#13E43C','#4430ED'],marker='D')
plt.legend(bbox_to_anchor=(0.5, 1.08), loc='upper center', fontsize='large', ncol=4,handletextpad=0.3,borderpad=0.2)                          
plt.ylabel('Accuracy (%)', fontsize='large')
plt.xlabel('Speed Range (km/h)', fontsize='large')
plt.tight_layout()
plt.grid(True) 
fig.figure.savefig('./Figures/RB_SpeedVariation.png',dpi=900) 

#=============================================================================#
###################### Robustness object variation #############################
#=============================================================================#
histo={'M1':[99.01, 93.74, 97.55, 95.72],'M2':[81.06, 76.11, 76.43, 76.21],'M3':[56.89, 75.27, 91.01, 96.66],'M4':[27.04, 97.01, 86.24, 61.72]}
df=pd.DataFrame.from_dict(histo, orient='index',columns=['ResNet','CWT+ResNet','HT+ResNet','GBDT'])
fig=df.plot(kind='line',color=['#E41352','#736D6D','#13E43C','#4430ED'],marker='D')
plt.legend(bbox_to_anchor=(0.5, 1.08), loc='upper center', fontsize='large', ncol=4,handletextpad=0.3,borderpad=0.2)                                                    
plt.ylabel('Accuracy (%)', fontsize='large')
plt.xlabel('Dataset', fontsize='large')
plt.tight_layout()
plt.grid(True) 
fig.figure.savefig('./Figures/RB_ObjectVariation.png',dpi=900) 

#=============================================================================#
#################### Robustness interference variation ########################
#=============================================================================#
histo={'T0_clean':[98.95, 98.20, 99.06, 98.13],'T1_WFbad':[89.70, 80.77, 69.40, 54.55],'T2_osc':[99.01, 98.07, 98.29, 53.09],'T3_peak':[98.42, 98.61, 99.42, 54.23],'T4_per':[98.83, 94.70, 96.28, 52.02]}
df=pd.DataFrame.from_dict(histo, orient='index',columns=['ResNet','CWT+ResNet','HT+ResNet','GBDT'])
fig=df.plot(kind='line',color=['#E41352','#736D6D','#13E43C','#4430ED'],marker='D')
plt.legend(bbox_to_anchor=(0.5, 1.08), loc='upper center', fontsize='large', ncol=4,handletextpad=0.3,borderpad=0.2)                                                    
plt.ylabel('Accuracy (%)', fontsize='large')
plt.xlabel('Signal interferences', fontsize='large')
plt.tight_layout()
plt.grid(True)
fig.figure.savefig('./Figures/RB_interfVariation.png',dpi=900) 

#=============================================================================#
########################### synthetic data samples ############################
#=============================================================================#
raw = a_ESZUEG_Bad_v15v25[57]
time = np.linspace(0,2.048,num=len(raw),endpoint=False) 
fig,ax=plt.subplots()
ax.plot(time,raw,color='#0000fb')
plt.xlim((0,2.048))
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.savefig('./Figures/a_aug_ESZUEG.png',dpi=600) 

##
raw = a_syn_Bad_v20[11]
time = np.linspace(0,2.048,num=len(raw),endpoint=False) 
fig,ax=plt.subplots()
ax.plot(time,raw,color='#0000fb')
plt.xlim((0,2.048))
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.savefig('./Figures/a_aug_FWDBA.png',dpi=600) 


##
FolderPath = os.path.abspath(os.path.join(os.getcwd(), "./Data/Augmentation/output_data/"))
sim_WF=np.load(FolderPath+'/a_sim_Bad_v20.npy')           
raw = sim_WF[11]
time = np.linspace(0,2.048,num=len(raw),endpoint=False) 
fig,ax=plt.subplots()
ax.plot(time,raw,color='#0000fb')
plt.xlim((0,2.048))
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.savefig('./Figures/a_aug_SIM.png',dpi=600) 

##
raw = X[1,:,0]
time = np.linspace(0,2.048,num=len(raw),endpoint=False) 
fig,ax=plt.subplots()
ax.plot(time,raw,color='#0000fb')
plt.xlim((0,2.048))
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.savefig('./Figures/a_aug_cDCGAN.png',dpi=600) 

##
raw = X[2,:,0]
time = np.linspace(0,2.048,num=len(raw),endpoint=False) 
fig,ax=plt.subplots()
ax.plot(time,raw,color='#0000fb')
plt.xlim((0,2.048))
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.savefig('./Figures/a_aug_SIM-GAN.png',dpi=600) 

##
FolderPath = os.path.abspath(os.path.join(os.getcwd(), "./Data/Augmentation/output_data/"))
sim_WF=np.load(FolderPath+'/a_sim_Bad_v20_30mm.npy')           
raw = sim_WF[1,:]
time = np.linspace(0,2.048,num=len(raw),endpoint=False) 
fig,ax=plt.subplots()
ax.plot(time,raw,color='#0000fb')
plt.xlim((0,2.048))
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.savefig('./Figures/a_aug_sim_2x30mm.png',dpi=600) 
#
FolderPath = os.path.abspath(os.path.join(os.getcwd(), "./Data/Augmentation/output_data/"))
syn_WF=np.load(FolderPath+'/a_syn_Bad_v20_30mm.npy')           
raw = syn_WF[16,:]
time = np.linspace(0,2.048,num=len(raw),endpoint=False) 
fig,ax=plt.subplots()
ax.plot(time,raw,color='#0000fb')
plt.xlim((0,2.048))
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.savefig('./Figures/a_aug_FWFSA_2x30mm.png',dpi=600) 
#
FolderPath = os.path.abspath(os.path.join(os.getcwd(), "./Data/Augmentation/output_data/"))
syn_WF=np.load(FolderPath+'/a_syn_Bad_v20v50_simgan_30mm.npy')           
raw = syn_WF[12,:]
time = np.linspace(0,2.048,num=len(raw),endpoint=False) 
fig,ax=plt.subplots()
ax.plot(time,raw,color='#0000fb')
plt.xlim((0,2.048))
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.savefig('./Figures/a_aug_SIM-GAN_2x30mm.png',dpi=600) 
#
FolderPath = os.path.abspath(os.path.join(os.getcwd(), "./Data/Augmentation/output_data/"))
syn_WF=np.load(FolderPath+'/a_syn_Bad_v15v45_cdcgan.npy')           
raw = syn_WF[1,:]
time = np.linspace(0,2.048,num=len(raw),endpoint=False) 
fig,ax=plt.subplots()
ax.plot(time,raw,color='#0000fb')
plt.xlim((0,2.048))
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.savefig('./Figures/a_aug_cDCGAN_2x30mm.png',dpi=600)
#=============================================================================#
################ Robustness enhancement: speed variation ######################
#=============================================================================#
gray   = '#A5A5A5'
red    = '#ED7D31'
yellow = '#FBBE00'
green  = '#70AD47'
blue   = '#75ABDC'
black  = '#000000'

histo={'v55v105':[99.85, 98.95, 99.46, 99.91, 99.94],
       'v45v55': [98.42, 97.36, 98.45, 97.91, 98.21],
       'v35v45': [96.88, 96.56, 98.43, 98.75, 98.85],
       'v25v35': [80.95, 88.59, 90.99, 88.13, 85.07],
       'v15v25': [58.00, 72.91, 67.73, 60.36, 57.12]}
df=pd.DataFrame.from_dict(histo, orient='index',columns=['orig','MBS-FWFSA','SIM','SIM-GAN','cCDGAN'])
fig=df.plot(kind='bar', color=[blue,red,yellow,green,gray], figsize=(12, 6))
for p in fig.patches:
    value = p.get_height()
    fig.annotate("{:.1f}".format(value), (p.get_x() * 1.005, p.get_height() * 1.005), fontsize = 8)
plt.legend(bbox_to_anchor=(0.5, 1.08), loc='upper center', fontsize='large', ncol=5,handletextpad=0.3,borderpad=0.2)                          
plt.xticks(rotation=0)
plt.ylabel('Accuracy in %', fontsize='large')
plt.tight_layout()
fig.figure.savefig('./Figures/RE_SpeedVariation.png',dpi=900)

#=============================================================================#
################ Robustness enhancement: Object variation #####################
#=============================================================================#
histo={'M1':[99.01, 97.72, 98.17, 98.13, 97.54],
       'M2': [81.06, 86.33, 76.48, 83.45, 80.82],
       'M3': [56.89, 79.62, 70.26, 69.77, 63.38],
       'M4': [27.04, 74.52, 58.91, 62.22, 50.17]}
df=pd.DataFrame.from_dict(histo, orient='index',columns=['orig','MBS-FWFSA','SIM','SIM-GAN','cCDGAN'])
fig=df.plot(kind='bar', color=[blue,red,yellow,green,gray], figsize=(12, 6))
for p in fig.patches:
    value = p.get_height()
    fig.annotate("{:.1f}".format(value), (p.get_x() * 1.005, p.get_height() * 1.005), fontsize = 8)
plt.legend(bbox_to_anchor=(0.5, 1.08), loc='upper center', fontsize='large', ncol=5,handletextpad=0.3,borderpad=0.2)                          
plt.xticks(rotation=0)
plt.ylabel('Accuracy in %', fontsize='large')
plt.tight_layout()
fig.figure.savefig('./Figures/RE_ObjectVariation.png',dpi=900)

#=============================================================================#
############## Robustness enhancement: interference variation #################
#=============================================================================#
# GBDT
histo={'T0_clean':[98.13, 98.69, 97.72, 97.75, 97.83],
       'T1_WFbad':[54.55, 54.85, 57.51, 57.51, 50.86],
       'T2_osc':  [53.09, 55.25, 52.71, 44.03, 61.44],
       'T3_peak': [54.23, 57.03, 52.82, 51.39, 59.34],
       'T4_per':  [52.02, 52.15, 49.76, 51.50, 54.29]}
df=pd.DataFrame.from_dict(histo, orient='index',columns=['orig','MBS-FWFSA','SIM','SIM-GAN','cCDGAN'])
fig=df.plot(kind='bar', color=[blue,red,yellow,green,gray], figsize=(12, 6))
for p in fig.patches:
    value = p.get_height()
    fig.annotate("{:.1f}".format(value), (p.get_x() * 1.005, p.get_height() * 1.005), fontsize = 8)
plt.legend(bbox_to_anchor=(0.5, 1.08), loc='upper center', fontsize='large', ncol=5,handletextpad=0.3,borderpad=0.2)                          
plt.xticks(rotation=0)
plt.ylabel('Accuracy in %', fontsize='large')
plt.tight_layout()
fig.figure.savefig('./Figures/RE_InterferenceVariation_GBDT.png',dpi=900)

# HT_ResNet
histo={'T0_clean':[99.06, 99.10, 99.03, 98.91, 99.14],
       'T1_WFbad':[69.40, 76.91, 73.18, 81.24, 74.72],
       'T2_osc':  [98.29, 97.51, 98.95, 91.82, 97.57],
       'T3_peak': [99.42, 98.42, 99.30, 96.18, 98.56],
       'T4_per':  [96.28, 97.09, 98.30, 88.02, 94.37]}
df=pd.DataFrame.from_dict(histo, orient='index',columns=['orig','MBS-FWFSA','SIM','SIM-GAN','cCDGAN'])
fig=df.plot(kind='bar', color=[blue,red,yellow,green,gray], figsize=(12, 6))
for p in fig.patches:
    value = p.get_height()
    fig.annotate("{:.1f}".format(value), (p.get_x() * 1.005, p.get_height() * 1.005), fontsize = 8)
plt.legend(bbox_to_anchor=(0.5, 1.08), loc='upper center', fontsize='large', ncol=5,handletextpad=0.3,borderpad=0.2)                          
plt.xticks(rotation=0)
plt.ylabel('Accuracy in %', fontsize='large')
plt.tight_layout()
fig.figure.savefig('./Figures/RE_InterferenceVariation_HTResnet.png',dpi=900)

#=============================================================================#
############## interference variation: feature plot ###########################
#=============================================================================#
from tensorflow.keras import models
# prepare the activation model
model.summary()
layer_names = []
for layer in model.layers:
    layer_names.append(layer.name) 
act_index=[position for position, name in enumerate(layer_names) if 'global_average_pooling1d' in name]  # index of activation layers
act_index=np.asarray(act_index)
actlayer_names=[layer_names[i] for i in act_index]
layer_outputs =[model.layers[i].output for i in act_index]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

## with data augmentation
rs = 25
x = np.vstack((x_train,x_test_0,x_test_1,x_test_2,x_test_3,x_train_aug))
indics_orig = list(range(len(x_train)))
indics_0 = list(range(indics_orig[-1]+1, 1+indics_orig[-1]+len(x_test_0)))
indics_1 = list(range(indics_0[-1]+1, 1+indics_0[-1]+len(x_test_1)))
indics_2 = list(range(indics_1[-1]+1, 1+indics_1[-1]+len(x_test_2)))
indics_3 = list(range(indics_2[-1]+1, 1+indics_2[-1]+len(x_test_3)))
indics_aug = list(range(indics_3[-1]+1, 1+indics_3[-1]+len(x_train_aug)))

# x_act = activation_model.predict(x) 
x_act = x[:,:,0]
# x_act = x
x_act = PCA(n_components=32, random_state=rs).fit_transform(x_act)
x_act_ft = TSNE(n_components=2, perplexity=39, random_state=rs).fit_transform(x_act)
x_act_ft_orig = x_act_ft[indics_orig]
x_act_ft_0 = x_act_ft[indics_0]
x_act_ft_1 = x_act_ft[indics_1]
x_act_ft_2 = x_act_ft[indics_2]
x_act_ft_3 = x_act_ft[indics_3]
x_act_ft_aug = x_act_ft[indics_aug]

x_act_ft_orig_Good = x_act_ft_orig[np.where(y_train[:,0]==0)]
x_act_ft_orig_Bad = x_act_ft_orig[np.where(y_train[:,0]==1)]

x_act_ft_aug_Good = x_act_ft_aug[np.where(y_train_aug[:,0]==0)]
x_act_ft_aug_Bad = x_act_ft_aug[np.where(y_train_aug[:,0]==1)]

# T0_clean vs Aug
size = 10
gray   = '#A5A5A5'
red    = '#fb4c00'
yellow = '#FFD861'
green  = '#70AD47'
blue   = '#75ABDC'
black  = '#000000'

plt.scatter(x=x_act_ft_orig_Good[:,0], y=x_act_ft_orig_Good[:,1],color=yellow,s=size)
plt.scatter(x=x_act_ft_orig_Bad[:,0], y=x_act_ft_orig_Bad[:,1],color=yellow,s=size, edgecolors=black)  

plt.scatter(x=x_act_ft_aug_Good[:,0], y=x_act_ft_aug_Good[:,1],color=red, s=size)
plt.scatter(x=x_act_ft_aug_Bad[:,0], y=x_act_ft_aug_Bad[:,1],color=red, s=size, edgecolors=black)  


# T0_clean vs T1
plt.scatter(x=x_act_ft_orig_Good[:,0], y=x_act_ft_orig_Good[:,1],color=yellow,s=size)
plt.scatter(x=x_act_ft_orig_Bad[:,0], y=x_act_ft_orig_Bad[:,1],color=yellow,s=size,edgecolors=black)  
plt.scatter(x=x_act_ft_0[:,0], y=x_act_ft_0[:,1],color=blue,s=size, edgecolors=black) 

# T0_clean vs T2
plt.scatter(x=x_act_ft_orig_Good[:,0], y=x_act_ft_orig_Good[:,1],color='#fbbe00',s=size)
plt.scatter(x=x_act_ft_orig_Bad[:,0], y=x_act_ft_orig_Bad[:,1],color='#fbbe00',s=size, edgecolors='#000000') 
plt.scatter(x=x_act_ft_1[:,0], y=x_act_ft_1[:,1],color='#0000fb',s=size)

# T0_clean vs T3
plt.scatter(x=x_act_ft_orig_Good[:,0], y=x_act_ft_orig_Good[:,1],color='#aaaaaa',s=10)
plt.scatter(x=x_act_ft_orig_Bad[:,0], y=x_act_ft_orig_Bad[:,1],color='#aaaaaa',s=10, edgecolors='#000000', marker='s') 
plt.scatter(x=x_act_ft_2[:,0], y=x_act_ft_2[:,1],color='#00fb00',s=10)

# T0_clean vs T4
plt.scatter(x=x_act_ft_orig_Good[:,0], y=x_act_ft_orig_Good[:,1],color='#aaaaaa',s=10)
plt.scatter(x=x_act_ft_orig_Bad[:,0], y=x_act_ft_orig_Bad[:,1],color='#aaaaaa',s=10, edgecolors='#000000', marker='s') 
plt.scatter(x=x_act_ft_3[:,0], y=x_act_ft_3[:,1],color='#fbfb00',s=10)
