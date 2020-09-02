import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import random
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import pandas as pd
import pywt
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.signal import hilbert
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from FeatureExtraction import iF, iFEn_2D, waveletpower

FigPath = os.path.abspath(os.path.join(os.getcwd(), "./Figure_scalogram_syn/"))
FolderPath1 = ['v20']
FolderPath2 = ['WF']
fs=500
dt = 1/fs
wavelet = 'cmor1.5-1.0'
scales = np.arange(1, 255)


for i in range(len(FolderPath1)):
    for j in range(len(FolderPath2)):
        varname_a1 = 'syn_'+ FolderPath2[j] + '_' +  FolderPath1[i]
        varname_feature1 = 'feature1_' + FolderPath2[j]+ '_' + FolderPath1[i]
        varname_feature2 = 'feature2_' + FolderPath2[j]+ '_' + FolderPath1[i]
        globals()[varname_feature1] = np.empty((0))
        globals()[varname_feature2] = np.empty((0))
        N=len(globals()[varname_a1])        
        for n in range(N):
            sample = globals()[varname_a1][n,:]
            [cfs, frequencies] = pywt.cwt(sample, scales, wavelet, 1/fs)
            power = (abs(cfs)) ** 2
            
            freqs = pywt.scale2frequency(wavelet,scales) / dt
            mask = frequencies < 501
            index = np.where(mask)
            time = np.linspace(0, 2, 2*fs)
            t, f = np.meshgrid(time, frequencies)
            
            globals()[varname_feature1] = np.append(globals()[varname_feature1],iFEn_2D(power,frequencies))
            globals()[varname_feature2] = np.append(globals()[varname_feature2],waveletpower(power,frequencies,scales))
            NpFilePath1=FigPath+'/'+varname_feature1+'.npy'
            NpFilePath2=FigPath+'/'+varname_feature2+'.npy'
            np.save(NpFilePath1,globals()[varname_feature1])
            np.save(NpFilePath2,globals()[varname_feature2])
            
            plt.close('all')
            fig, axContour = plt.subplots(figsize=(10, 8))
            axContour.pcolormesh(t[index], f[index], power[index])
            axContour.grid(True)
            axContour.set_title("Morlet scalogram")
            axContour.set_ylabel('Frequency')
            axContour.set_xlabel('Time')
            
            divider = make_axes_locatable(axContour)
            axTime = divider.append_axes("top", 2, pad=0.8)
            axTime.plot(time,sample)
            plt.xlim((0, 2))
            axTime.set_ylabel('normalised Acceleration')
            figname=varname_a1 + '['+str(n)+']'
            axTime.set_title(figname)
            axTime.grid(True)              
            FigFilePath=FigPath+'/'+figname+'.png'
            plt.savefig(FigFilePath)
###############################################################################
#FigPath = os.path.abspath(os.path.join(os.getcwd(), "./Figure_scalogram_syn1/"))
#FolderPath1 = ['peak','per']
#fs=500
#dt = 1/fs
#wavelet = 'cmor1.5-1.0'
#scales = np.arange(1, 255)
#
#
#for i in range(len(FolderPath1)):
#    varname_a1 = 'x_syn_' +  FolderPath1[i]
#    varname_feature1 = 'feature1_'  + FolderPath1[i]
#    varname_feature2 = 'feature2_'  + FolderPath1[i]
#    globals()[varname_feature1] = np.empty((0))
#    globals()[varname_feature2] = np.empty((0))
#    N=len(globals()[varname_a1])        
#    for n in range(N):
#        sample = globals()[varname_a1][n,:]
#        [cfs, frequencies] = pywt.cwt(sample, scales, wavelet, 1/fs)
#        power = (abs(cfs)) ** 2
#        
#        freqs = pywt.scale2frequency(wavelet,scales) / dt
#        mask = frequencies < 501
#        index = np.where(mask)
#        time = np.linspace(0, 2, 2*fs)
#        t, f = np.meshgrid(time, frequencies)
#        
#        globals()[varname_feature1] = np.append(globals()[varname_feature1],iFEn_2D(power,frequencies))
#        globals()[varname_feature2] = np.append(globals()[varname_feature2],waveletpower(power,frequencies,scales))
#        NpFilePath1=FigPath+'/'+varname_feature1+'.npy'
#        NpFilePath2=FigPath+'/'+varname_feature2+'.npy'
#        np.save(NpFilePath1,globals()[varname_feature1])
#        np.save(NpFilePath2,globals()[varname_feature2])
#        
#        plt.close('all')
#        fig, axContour = plt.subplots(figsize=(10, 8))
#        axContour.pcolormesh(t[index], f[index], power[index])
#        axContour.grid(True)
#        axContour.set_title("Morlet scalogram")
#        axContour.set_ylabel('Frequency')
#        axContour.set_xlabel('Time')
#        
#        divider = make_axes_locatable(axContour)
#        axTime = divider.append_axes("top", 2, pad=0.8)
#        axTime.plot(time,sample)
#        plt.xlim((0, 2))
#        axTime.set_ylabel('normalised Acceleration')
#        figname=varname_a1 + '['+str(n)+']'
#        axTime.set_title(figname)
#        axTime.grid(True)              
#        FigFilePath=FigPath+'/'+figname+'.png'
#        plt.savefig(FigFilePath)          