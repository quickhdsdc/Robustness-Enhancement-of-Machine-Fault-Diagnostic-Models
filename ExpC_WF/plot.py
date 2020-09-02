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
########################### Plot signals ######################################
#=============================================================================#
fig1,[[ax1, ax2],[ax3, ax4]]=plt.subplots(nrows=2, ncols=2, sharex=False)
fs=5000
t = np.linspace(0, 2, 2*fs)
ax1.plot(t,a_ESZUEG_v35,color='#0021D7')
ax2.plot(t,a_Bosch_v35, color='#0021D7')
fs=500
t = np.linspace(0, 2, 2*fs)               
ax3.plot(t,a_LEILA_v35,color='#0021D7')
ax4.plot(t,a_Entgleis_v35, color='#0021D7')

         

ax1.set_title('M1 20mm WF')
ax1.set_ylabel("Acceleration in m/s2")
ax2.set_title('M2 50mm WF')
ax3.set_title('M3 60mm WF')
ax3.set_ylabel("Acceleration (m/s2)")
ax3.set_xlabel("Time (s)")
ax4.set_title('M4 two 50mm WF')
ax4.set_xlabel("Time (s)")

plt.tight_layout()

fig1.savefig('./Figure/Wfsignals.png',dpi=600)

#=============================================================================#
########################### Faulty condition variation ########################
#=============================================================================#
histo={'20mm(M1)':[98.59, 99.23, 99.00, 94.15],'50mm(M2)':[73, 62.68, 54.75, 81.92],'60mm(M3)':[90.41, 96.06, 78.15, 91.86],'2-50mm(M4)':[71.19, 94.29, 86.18, 50.62]}
df=pd.DataFrame.from_dict(histo, orient='index',columns=['ResNet','WPT+ResNet','STFT+2DResNet','HT+GBDT'])
fig=df.plot(kind='line',color=['#E41352','#736D6D','#13E43C','#4430ED'],marker='D')
plt.legend(bbox_to_anchor=(0.5, 1.08), loc='upper center', fontsize='small', ncol=6,handletextpad=0.3,borderpad=0.2)                          
plt.ylabel('Accuracy (%)')
plt.xlabel('Faulty condition')
plt.tight_layout()
plt.grid(True) 
fig.figure.savefig('./Figure/FaultyCondition.png',dpi=600) 





   
