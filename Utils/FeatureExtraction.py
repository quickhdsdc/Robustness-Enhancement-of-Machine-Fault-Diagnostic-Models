###############################################################################
##########################  feature dunctions #################################
###############################################################################

#=============================================================================#
############################# import functions ################################
#=============================================================================#
import numpy as np
import math
import pywt
from scipy.signal import hilbert
from scipy import signal
from scipy.spatial.distance import euclidean
import pandas as pd
from numpy import dot, exp
from scipy.spatial.distance import cdist

###############################################################################
def fft_powerspectrum(data,fs,f_cutoff):
#    fft_size=next_power_of_2(len(data))
    fft_size=len(data)
    n = len(data) // fft_size * fft_size
    data_tmp = data[:n].reshape(fft_size)
    [b,a]=signal.butter(2,[200/(fs/2)],'lowpass')
    data_filter=signal.filtfilt(b, a, data_tmp)
    data_filter *= signal.hann(fft_size, sym=0)
    data_filter=data_filter-np.mean(data_filter)
    data_fft = np.abs(np.fft.rfft(data_filter)/fft_size)
    data_fft = data_fft[0:fft_size//2]
    freqs = np.fft.fftfreq(fft_size,1/fs)
    freqs = freqs[:fft_size//2]  
    fft_cutoff=data_fft[0:f_cutoff]
    Freq=freqs[0:f_cutoff]
    return fft_cutoff, Freq


def envelope_powerspectrum(data,fs,f_cutoff):
    #fft_size=next_power_of_2(len(data))
    fft_size=len(data)
    n = len(data) // fft_size * fft_size
    data_tmp = data[:n].reshape(fft_size)
    analytic_signal=hilbert(data_tmp)
    amplitude_envelope = np.abs(analytic_signal)
    [b,a]=signal.butter(2,[2/(fs/2),200/(fs/2)],'bandpass')
    amplitude_envelope_filter=signal.filtfilt(b, a, amplitude_envelope)
    amplitude_envelope_filter *= signal.hann(fft_size, sym=0)
    amplitude_envelope_filter=amplitude_envelope_filter-np.mean(amplitude_envelope_filter)
    envelope_fft = np.abs(np.fft.rfft(amplitude_envelope_filter)/fft_size)
    envelope_fft = envelope_fft[0:fft_size//2]
    freqs = np.fft.fftfreq(fft_size,1/fs)
    freqs = freqs[:fft_size//2]
    envelope_cutoff=envelope_fft[0:f_cutoff]
    Freq=freqs[0:f_cutoff]
    return envelope_cutoff, Freq

def cwt(data,fs,f_cutoff):
    dt = 1/fs
    wavelet = 'cmor1.5-1.0'
    scales = np.arange(1, 255)
    [cfs, frequencies] = pywt.cwt(data, scales, wavelet, 1/fs)
    power = (abs(cfs)) ** 2*10   
    freqs = pywt.scale2frequency(wavelet,scales) / dt
    mask = frequencies <= f_cutoff
    index = np.where(mask)
    time = np.linspace(0,len(data)/fs,num=len(data),endpoint=False)
    t, f = np.meshgrid(time, frequencies)      
    return t[index], f[index], power[index]


# def wpt(data,level):
#     level=level
#     datasize=np.shape(data)[0]
#     channel=pow(2,level)
#     wp_sample=pywt.WaveletPacket(data=data[0,:,0], wavelet='db4', mode='symmetric',maxlevel=level)
#     width=len(wp_sample['aaa'].data)
#     output=np.zeros((datasize,width,channel))
#     wp_index=[]
#     wp_index.append([node.path for node in wp_sample.get_level(level, 'natural')])
#     wp_index=wp_index[0]
#     for n in range(datasize):
#         sample=data[n,:,0]
#         wp=pywt.WaveletPacket(data=sample, wavelet='db4', mode='symmetric',maxlevel=level)
#         for i in range(channel):
#             output[n,:,i]=wp[wp_index[i]].data
#     return output

# def stft_powerspectrum(data,fs):
#     datasize=np.shape(data)[0]
#     f, t, Zxx = signal.stft(data[0,:,0], fs, nperseg=64)
#     width=len(t)
#     output=np.zeros((datasize,width,width,1))
#     for n in range(datasize):
#         sample=data[n,:,0]
#         f, t, Zxx = signal.stft(sample, fs, nperseg=64)
#         power=np.abs(Zxx[:width,:width])
#         output[n,:,:,0]=power
#     return output

def feature_normalize(dataset):
    mu = np.mean(dataset)
    sigma = np.std(dataset)
    temp = (dataset - mu)/sigma
    temp = temp/np.max(temp)
    return temp

#=============================================================================#
################### statistic features (Rauber et al, 2015) ###################
#=============================================================================#

### root mean square ###
def rms(data):
    data = data**2
    az_rms = data.sum()
    az_rms = az_rms/len(data)
    az_rms = math.sqrt(az_rms)
    return az_rms

### square root of the amplitude ### 

def sra(data):
    data = abs(data)**(1/2)
    az_sra = data.sum()
    az_sra = az_sra/len(data)
    az_sra = az_sra**2
    return az_sra
    
### kurtosis_value ###

def kv(data):
    mean = data.mean()
    std = data.std()
    data = (data-mean)/std
    data = data**4
    kv = data.sum()
    kv = kv/(len(data))
    return kv

### skewness value ###

def sv(data):
    mean = data.mean()
    std = data.std()
    data = (data-mean)/std
    data = data**3
    sv = data.sum()
    sv = sv/(len(data))
    return sv

### peak-to-peak value ###
    
def ppv(data):
    ppv = data.max() - data.min()
    return ppv

### crest factor ###
    
def cf(data):
    data_abs = abs(data)
    cf = data_abs.max()/rms(data)
    return cf

### impulse factor ###
    
def iF(data):
    data = abs(data)
    temp = data.sum()/len(data)
    iF = data.max()/temp
    return iF

### margin factor ###
    
def mf(data):
    data = abs(data)
    mf = data.max()/sra(data)
    return mf

### shape factor ###

def sf(data):
    data_abs = abs(data)
    temp = data_abs.sum()/len(data_abs)
    sf = rms(data)/temp
    return sf

### kurtosis factor ###

def kf(data):
    kf = kv(data)/(rms(data)**4)
    return kf

#=============================================================================#
######### feature extraction: CNN input, dataframe output #####################
#=============================================================================#
def ExtraFeatures(data,fs):
    a_sample=data
    # time domain features
    f_t1 = ppv(a_sample)
    f_t2 = rms(a_sample)
    f_t3 = kv(a_sample)
    f_t4 = iF(a_sample)
    f_t5 = mf(a_sample)
    f_t6 = sf(a_sample)
    f_t7 = cf(a_sample)
    # envelope spectrum features
    a_envelope,Freq=envelope_powerspectrum(a_sample,fs,512)
    f_e1 = ppv(a_envelope)
    f_e2 = rms(a_envelope)
    f_e3 = kv(a_envelope)
    f_e4 = iF(a_envelope)
    f_e5 = mf(a_envelope)
    f_e6 = sf(a_envelope)
    f_e7 = cf(a_envelope)        
    # scale-averaged wavelet spectrum features
    t, f, power = cwt(a_sample,5000,600)
    a_cwt = np.average(power,axis=0)
    f_c1 = ppv(a_cwt)
    f_c2 = rms(a_cwt)
    f_c3 = kv(a_cwt)
    f_c4 = iF(a_cwt)
    f_c5 = mf(a_cwt)
    f_c6 = sf(a_cwt)
    f_c7 = cf(a_cwt)        
            
    features=np.array([f_t1,f_t2,f_t3,f_t4,f_t5,f_t6,f_t7,
              f_e1,f_e2,f_e3,f_e4,f_e5,f_e6,f_e7,
              f_c1,f_c2,f_c3,f_c4,f_c5,f_c6,f_c7])
    
    return features


def ExtraFeatures_df(data,fs):
    df_test = pd.DataFrame()
    data.shape
    data = data.reshape((data.shape[0],data.shape[1]))
    for n in range(len(data)):
        a_sample=data[n,:]
        # time domain features
        f_t1 = ppv(a_sample)
        f_t2 = rms(a_sample)
        f_t3 = kv(a_sample)
        f_t4 = iF(a_sample)
        f_t5 = mf(a_sample)
        f_t6 = sf(a_sample)
        f_t7 = cf(a_sample)
        # envelope spectrum features
        a_envelope,Freq=envelope_powerspectrum(a_sample,fs,256)
        f_e1 = ppv(a_envelope)
        f_e2 = rms(a_envelope)
        f_e3 = kv(a_envelope)
        f_e4 = iF(a_envelope)
        f_e5 = mf(a_envelope)
        f_e6 = sf(a_envelope)
        f_e7 = cf(a_envelope)        
        # scale-averaged wavelet spectrum features
        t, f, power = cwt(a_sample,5000,600)
        a_cwt = np.average(power,axis=0)
        f_c1 = ppv(a_cwt)
        f_c2 = rms(a_cwt)
        f_c3 = kv(a_cwt)
        f_c4 = iF(a_cwt)
        f_c5 = mf(a_cwt)
        f_c6 = sf(a_cwt)
        f_c7 = cf(a_cwt)        
            
        features=[f_t1,f_t2,f_t3,f_t4,f_t5,f_t6,f_t7,
                  f_e1,f_e2,f_e3,f_e4,f_e5,f_e6,f_e7,
                  f_c1,f_c2,f_c3,f_c4,f_c5,f_c6,f_c7]
        df_features=pd.DataFrame(features,index=['f_t1','f_t2','f_t3','f_t4','f_t5','f_t6','f_t7',
                                                 'f_e1','f_e2','f_e3','f_e4','f_e5','f_e6','f_e7',
                                                 'f_c1','f_c2','f_c3','f_c4','f_c5','f_c6','f_c7'])  
        df_features=pd.DataFrame.transpose(df_features)
        df_test=df_test.append(df_features)
        print('progress: ' + str(round(100*n/len(data))) + '%')
    df_output=(df_test-df_test.mean())/df_test.std()
    return df_output