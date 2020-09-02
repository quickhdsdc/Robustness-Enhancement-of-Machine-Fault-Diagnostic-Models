import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from FeatureExtraction import iF, iFEn_2D, waveletpower
import matplotlib.gridspec as gridspec
from keras import models
from keras.models import load_model
from sklearn.metrics import roc_curve
from sklearn import metrics
from FeatureExtraction import envelope_powerspectrum, fft_powerspectrum, wpt,stft_powerspectrum,ExtraFeatures,LSSVM
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
import math
import seaborn as sns
from sklearn import preprocessing
import tensorflow as tf
import keras.backend as K
from keras.utils import plot_model
from Mish import Mish, Relu6, Hswish
#=============================================================================#
############################### speed variation ###############################
#=============================================================================#
histo={'15-55':[96.33, 93.51, 97.36, 89.90],'55-65':[97.29, 87.32, 96.15, 94.16],'65-75':[95.38, 63.10, 82.42, 95.27],'75-85':[95.38, 58.23, 71.27, 97.10],'85-95':[89.09, 45.96, 65.46, 96.85]}
df=pd.DataFrame.from_dict(histo, orient='index',columns=['ResNet','WPT+ResNet','STFT+2DResNet','HT+GBDT'])
fig=df.plot(kind='line',color=['#E41352','#736D6D','#13E43C','#4430ED'],marker='D')
plt.legend(bbox_to_anchor=(0.5, 1.08), loc='upper center', fontsize='small', ncol=6,handletextpad=0.3,borderpad=0.2)                          
plt.ylabel('balanced Accuracy (%)')
plt.xlabel('Speed range (km/h)')
plt.tight_layout()
plt.grid(True) 
fig.figure.savefig('./plot/SpeedVariation1.png',dpi=600) 

histo={'55-95':[99.50, 98.92, 98.28, 97.59],'45-55':[94.38, 93.61, 93.39, 93.65],'35-45':[96.41, 95.78, 94.71, 95.77],'25-35':[86.47, 82.77, 83.73, 92.12],'15-25':[60.61, 56.78, 60.55, 79.93]}
df=pd.DataFrame.from_dict(histo, orient='index',columns=['ResNet','WPT+ResNet','STFT+2DResNet','HT+GBDT'])
fig=df.plot(kind='line',color=['#E41352','#736D6D','#13E43C','#4430ED'],marker='D')
plt.legend(bbox_to_anchor=(0.5, 1.08), loc='upper center', fontsize='small', ncol=6,handletextpad=0.3,borderpad=0.2)                          
plt.ylabel('balanced Accuracy (%)')
plt.xlabel('Speed range (km/h)')
plt.tight_layout()
plt.grid(True) 
fig.figure.savefig('./figures/SpeedVariation2.png',dpi=600) 

# visualisation
t = np.linspace(0, 2, len(a_v20))
fs=500
fig,ax=plt.subplots()
ax.plot(t,a_v20,color='#0021D7')
plt.xlim((0,2))

fig,ax=plt.subplots()
waveletplot(a_v100,1000,500,ax)

fig,ax=plt.subplots()
envelopeplot(a_v20,1000,200,ax)

data=wpt(np.reshape(a_v20,(1,1000,1)),3)
plt.plot(data[0,:,4], color='#0021D7')
         
f, t, Zxx = signal.stft(a_v100, 500, nperseg=64)
width=len(t) 
power=np.abs(Zxx[:width,:width])
plt.pcolormesh(t, f[:width], power)
plt.show()    

   
