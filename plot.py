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
################## plot sim, synthetic, augmentation data #####################
#=============================================================================#
# compare axlebox bogie and carbody acceleration
fs=500
t = np.linspace(0, 2, fs)
plt.plot(x_test_per[5],color='#0021D7')
plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    left=False,
    top=False,         # ticks along the top edge are off
    labelbottom=False,
    labelleft=False) # labels along the bottom edge are off
plt.show()

plt.plot(signal.resample(sim_WF_v50[0],1000),color='#0021D7')
plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    left=False,
    top=False,         # ticks along the top edge are off
    labelbottom=False,
    labelleft=False) # labels along the bottom edge are off
plt.show()

plt.plot(x_syn_per[3],color='#0021D7')
plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    left=False,
    top=False,         # ticks along the top edge are off
    labelbottom=False,
    labelleft=False) # labels along the bottom edge are off
plt.show()


temp=SNR_Noise_1D(sim_WF_v50[0],10)
temp_rev=temp+x_syn_per[3]*1.5

plt.plot(temp_rev,color='#0021D7')
plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    left=False,
    top=False,         # ticks along the top edge are off
    labelbottom=False,
    labelleft=False) # labels along the bottom edge are off
plt.show()



plt.plot(sim_WF_v50_rev[450],color='#0021D7')
plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    left=False,
    top=False,         # ticks along the top edge are off
    labelbottom=False,
    labelleft=False) # labels along the bottom edge are off
plt.show()

plt.plot(syn_WF_v50[1],color='#0021D7')
plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    left=False,
    top=False,         # ticks along the top edge are off
    labelbottom=False,
    labelleft=False) # labels along the bottom edge are off
plt.show()

plt.plot(x_test[76,:,0],color='#0021D7')
plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    left=False,
    top=False,         # ticks along the top edge are off
    labelbottom=False,
    labelleft=False) # labels along the bottom edge are off
plt.show()
