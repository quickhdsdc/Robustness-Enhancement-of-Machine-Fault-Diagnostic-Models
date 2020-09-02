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
########################### Track condition variation ###############################
#=============================================================================#
histo={'T0':[99.16, 98.68, 99.29, 98.73],'T1':[81.37, 75.2, 88.46, 71.45],'T2':[98.62, 98.29, 99.28, 36.8],'T3':[78.97, 88.20, 94.27, 39.54],'T4':[74.58, 80.00, 85.94, 37.99]}
df=pd.DataFrame.from_dict(histo, orient='index',columns=['ResNet','WPT+ResNet','STFT+2DResNet','HT+GBDT'])
fig=df.plot(kind='line',color=['#E41352','#736D6D','#13E43C','#4430ED'],marker='D')
plt.legend(bbox_to_anchor=(0.5, 1.08), loc='upper center', fontsize='small', ncol=6,handletextpad=0.3,borderpad=0.2)                          
plt.ylabel('Accuracy (%)')
plt.xlabel('Faulty condition')
plt.tight_layout()
plt.grid(True) 
fig.figure.savefig('./Figure/InterferenceVariation.png',dpi=600) 


