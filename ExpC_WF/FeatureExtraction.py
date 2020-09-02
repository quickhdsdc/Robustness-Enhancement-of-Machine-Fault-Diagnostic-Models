###############################################################################
##########################  feature dunctions #################################
###############################################################################

#=============================================================================#
############################# import functions ################################
#=============================================================================#
import numpy as np
import math
from fastdtw import fastdtw
import pywt
from scipy.signal import hilbert
from scipy import signal
from scipy.spatial.distance import euclidean
import pandas as pd
from numpy import dot, exp
from scipy.spatial.distance import cdist
###############################################################################
#def next_power_of_2(x):  
#    return 1 if x == 0 else 2**(x - 1).bit_length()

def fft_powerspectrum(data,fs,f_cutoff):
#    fft_size=next_power_of_2(len(data))
    fft_size=len(data)
    n = len(data) // fft_size * fft_size
    data_tmp = data[:n].reshape(fft_size)
    [b,a]=signal.butter(2,[200/(fs/2)],'lowpass')
    data_filter=signal.filtfilt(b, a, data_tmp)
    data_filter *= signal.hann(fft_size, sym=0)
    data_filter=data_filter-np.mean(data_filter)
#    data_filter=data_tmp-np.mean(data_tmp)    
    data_fft = np.abs(np.fft.rfft(data_filter)/fft_size)
    data_fft = data_fft[0:fft_size//2]
    freqs = np.fft.fftfreq(fft_size,1/fs)
    freqs = freqs[:fft_size//2]  
    fft_cutoff=data_fft[0:f_cutoff]
    Freq=freqs[0:f_cutoff]
    #plt.plot(Freq,fft_200Hz)
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
#    amplitude_envelope_filter=amplitude_envelope-np.mean(amplitude_envelope)
    envelope_fft = np.abs(np.fft.rfft(amplitude_envelope_filter)/fft_size)
    envelope_fft = envelope_fft[0:fft_size//2]
    freqs = np.fft.fftfreq(fft_size,1/fs)
    freqs = freqs[:fft_size//2]
    envelope_cutoff=envelope_fft[0:f_cutoff]
    Freq=freqs[0:f_cutoff]
    #plt.plot(Freq, envelope_200Hz)
#    plt.plot(time, amplitude_envelope, 'k', time, amplitude_envelope_filter, 'r')
#    fig = plt.figure()
#    ax0 = fig.add_subplot(211)
#    ax0.plot(time, data, label='signal')
#    ax0.plot(time, amplitude_envelope, label='envelope')
#    ax0.set_xlabel("time in seconds")
#    ax0.legend()
    return envelope_cutoff, Freq

def wpt(data,level):
    level=level
    datasize=np.shape(data)[0]
    channel=pow(2,level)
    wp_sample=pywt.WaveletPacket(data=data[0,:,0], wavelet='db4', mode='symmetric',maxlevel=level)
    width=len(wp_sample['aaa'].data)
    output=np.zeros((datasize,width,channel))
    wp_index=[]
    wp_index.append([node.path for node in wp_sample.get_level(level, 'natural')])
    wp_index=wp_index[0]
    for n in range(datasize):
        sample=data[n,:,0]
        wp=pywt.WaveletPacket(data=sample, wavelet='db4', mode='symmetric',maxlevel=level)
        for i in range(channel):
            output[n,:,i]=wp[wp_index[i]].data
    return output

def stft_powerspectrum(data,fs):
    datasize=np.shape(data)[0]
    f, t, Zxx = signal.stft(data[0,:,0], fs, nperseg=256)
#    f, t, Zxx = signal.stft(data[0,:,0], fs, nperseg=64)    
    width=len(t)
    output=np.zeros((datasize,width,width,1))
    for n in range(datasize):
        sample=data[n,:,0]
        f, t, Zxx = signal.stft(sample, fs, nperseg=256)        
#        f, t, Zxx = signal.stft(sample, fs, nperseg=64)
        power=np.abs(Zxx[:width,:width])
        output[n,:,:,0]=power
#        plt.pcolormesh(t, f[:width], power)
#        plt.title('STFT Magnitude')
#        plt.ylabel('Frequency [Hz]')
#        plt.xlabel('Time [sec]')
#        plt.show()
    return output


def feature_normalize(dataset):
    mu = np.mean(dataset)
    sigma = np.std(dataset)
    temp = (dataset - mu)/sigma
    temp = temp/np.max(temp)
    return temp

def evaluate_IMF(imf,ref):
    imf_spectrum=envelope_powerspectrum(imf,fs,50)   
    similarity,path=fastdtw(feature_normalize(imf_spectrum),ref,dist=euclidean)
    return similarity
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
    
# extract statistical features for every data frame in work space

#alldfs = [var for var in dir() if isinstance(eval(var), pd.core.frame.DataFrame)]

#for i in range(len(alldfs)):
#    df = eval(alldfs[i])
#    df = df['a_z']
class LSSVM:
    'Class that implements the Least-Squares Support Vector Machine.'
   
    def __init__(self, gamma=1, kernel='rbf', **kernel_params): 
        self.gamma = gamma
        
        self.x        = None
        self.y        = None
        self.y_labels = None
        
        # model params
        self.alpha = None
        self.b     = None
        
        self.kernel = LSSVM.get_kernel(kernel, **kernel_params)
        
           
    @staticmethod
    def get_kernel(name, **params):
        
        def linear(x_i, x_j):                           
            return dot(x_i, x_j.T)
        
        def poly(x_i, x_j, d=params.get('d',3)):        
            return ( dot(x_i, x_j.T) + 1 )**d
        
        def rbf(x_i, x_j, sigma=params.get('sigma',1)):
            if x_i.ndim==x_i.ndim and x_i.ndim==2: # both matrices
                return exp( -cdist(x_i,x_j)**2 / sigma**2 )
            
            else: # both vectors or a vector and a matrix
                return exp( -( dot(x_i,x_i.T) + dot(x_j,x_j.T)- 2*dot(x_i,x_j) ) / sigma**2 )
#             temp = x_i.T - X
#             return exp( -dot(temp.temp) / sigma**2 )
                
        kernels = {'linear': linear, 'poly': poly, 'rbf': rbf}
                
        if kernels.get(name) is None: 
            raise KeyError("Kernel '{}' is not defined, try one in the list: {}.".format(
                name, list(kernels.keys())))
        else: return kernels[name]
        
    
    def opt_params(self, X, y_values):
        sigma = np.multiply( y_values*y_values.T, self.kernel(X,X) )

        A_cross = np.linalg.pinv(np.block([
            [0,                           y_values.T                   ],
            [y_values,   sigma + self.gamma**-1 * np.eye(len(y_values))]
        ]))

        B = np.array([0]+[1]*len(y_values))

        solution = dot(A_cross, B)
        b     = solution[0]
        alpha = solution[1:]
        
        return (b, alpha)
            
    
    def fit(self, X, Y, verboses=0):
        self.x = X
        self.y = Y
        self.y_labels = np.unique(Y, axis=0)
        
        if len(self.y_labels)==2: # binary classification
            # converting to -1/+1
            y_values = np.where(
                (Y == self.y_labels[0]).all(axis=1)
                ,-1,+1)[:,np.newaxis] # making it a column vector
            
            self.b, self.alpha = self.opt_params(X, y_values)
        
        else: # multiclass classification
              # ONE-VS-ALL APPROACH
            n_classes = len(self.y_labels)
            self.b     = np.zeros(n_classes)
            self.alpha = np.zeros((n_classes, len(Y)))
            for i in range(n_classes):
                # converting to +1 for the desired class and -1 for all other classes
                y_values = np.where(
                    (Y == self.y_labels[i]).all(axis=1)
                    ,+1,-1)[:,np.newaxis] # making it a column vector
  
                self.b[i], self.alpha[i] = self.opt_params(X, y_values)

        
    def predict(self, X):
        K = self.kernel(self.x, X)
        
        if len(self.y_labels)==2: # binary classification
            y_values = np.where(
                (self.y == self.y_labels[0]).all(axis=1),
                -1,+1)[:,np.newaxis] # making it a column vector

            Y = np.sign( dot( np.multiply(self.alpha, y_values.flatten()), K ) + self.b)
            
            y_pred_labels = np.where(Y==-1, self.y_labels[0], 
                                     self.y_labels[1])
        
        else: # multiclass classification, ONE-VS-ALL APPROACH
            Y = np.zeros((len(self.y_labels), len(X)))
            for i in range(len(self.y_labels)):
                y_values = np.where(
                    (self.y == self.y_labels[i]).all(axis=1),
                    +1, -1)[:,np.newaxis] # making it a column vector
                Y[i] = dot( np.multiply(self.alpha[i], y_values.flatten()), K ) + self.b[i] # no sign function applied
            
            predictions = np.argmax(Y, axis=0)
            y_pred_labels = np.array([self.y_labels[i] for i in predictions])
            
        return y_pred_labels
#=============================================================================#
#### faulty frequency feature in envelope spectrum (Rauber et al, 2015) #######
#=============================================================================#
### narrouw band RMS
  
#def NbRMS(data,v,fs,a_ref):
def NbRMS(data,v,fs):
#    envelope_ref = envelope_powerspectrum(a_ref,fs,200)
#    indics_peaks,temp=signal.find_peaks(envelope_ref,threshold=0.1)
#    w12=envelope_ref[indics_peaks[0]]/envelope_ref[indics_peaks[1]]
#    w23=envelope_ref[indics_peaks[1]]/envelope_ref[indics_peaks[2]]
#    w34=envelope_ref[indics_peaks[2]]/envelope_ref[indics_peaks[3]]
#    w45=envelope_ref[indics_peaks[3]]/envelope_ref[indics_peaks[4]]
#    w56=envelope_ref[indics_peaks[4]]/envelope_ref[indics_peaks[5]]  
#    w1=1
#    w2=w1/w12
#    w3=w2/w23
#    w4=w3/w34
#    w5=w4/w45  
#    w6=w5/w56    
    envelope=envelope_powerspectrum(data,fs,200)   
    f_h1=v/3.6/np.pi/np.mean([0.92,0.84])                                    # new diameter 920 mm; minimum diameter 840 mm
    f_h2=2*f_h1
    f_h3=3*f_h1
    f_h4=4*f_h1
    f_h5=5*f_h1
    f_h6=6*f_h1    
    freqs = np.fft.fftfreq(len(data),1/fs)
    freqs = freqs[1:len(freqs)//2]
    res=freqs[1]-freqs[0]
    f_band=res
    RMS=0
    for i in range(1,7):
        VarName = 'f_h'+str(i)       
        mask = (freqs > locals()[VarName]-f_band) & (freqs < locals()[VarName]+f_band)                             
        index = np.where(mask)
#        VarName1 = 'w'+str(i)   
#        RMS=RMS+rms(envelope[index])*locals()[VarName1]
        RMS=RMS+rms(envelope[index])
    return RMS

#=============================================================================#
#################### Energy-to-shannon entropy for IMFs #######################
#=============================================================================#
def EG2EP(data):                                                               # input "data" is a list of IMF; 
#    S=[]                                                                      # decomposition methods could be wavelet,EMD, VMD, EWT
#    Etot=[]
    S=0
    Etot=0
    for d in data:
        E=d**2
        P=E/np.sum(E)
        S+=-np.sum(P*np.log(P))
        Etot+=np.sum(E)
#        S.append(-np.sum(P*np.log(P)))
#        Etot.append(np.sum(E))
#    ratio=np.asarray(Etot)/np.asarray(S)
    ratio=Etot/S
    return ratio

#=============================================================================#
############### Similarity in envelope spectrum for IMFs ######################
#=============================================================================#
def ST(data,fs,a_ref):                                                           # input "data" is a list of IMF
    IMF_norm=np.apply_along_axis(feature_normalize,axis=1,arr=data)            # decomposition methods could be wavelet,EMD, VMD, EWT
    sample_similarity=[]
    reference=feature_normalize(envelope_powerspectrum(a_ref,fs,50))
    for n, imf in enumerate(IMF_norm):
        imf_spectrum=envelope_powerspectrum(imf,fs,50)
        imf_spectrum_norm=feature_normalize(imf_spectrum)
        similarity,path=fastdtw(imf_spectrum_norm,reference,dist=euclidean)
    #    similarity=euclidean(imf_spectrum_norm,reference)
        sample_similarity.append(similarity)
    indices=np.argsort(sample_similarity)[0:3]
    IMF_rec=np.zeros(np.shape(data[0]))
    for i in indices:
        IMF_rec=IMF_rec+data[i]  
    return evaluate_IMF(IMF_rec,reference)    

#=============================================================================#
#################### features for time-frequency spectrum #####################
#=============================================================================#
def iFEn_2D(data,freq):                                                       # input data is 2D time-frequency power spectrum (each row is the time domain in one frequency/scala)
    S=0                                                                        # methods could be continoues wavelet transform, STFT, Wigner distribution.
    Etot=0
    mask = freq < 500
    index = np.where(mask)[0]
    iF_row=[]
    for i in index:
        iF_row.append(iF(data[i]))
    E=np.asarray(iF_row)**2
    P=E/np.sum(E)
    S=-np.sum(P*np.log(P))
    Etot=np.sum(E)
    ratio=Etot/S
    return ratio

def waveletpower(data,freq,scales):                                                   # input data is 2D time-frequency power spectrum (each row is the time domain in one frequency/scala)
    Etot=0                                                                        # scale-averaged wavelet power
    mask = freq < 500
    index = np.where(mask)[0]
    power_allscale=[]
    for i in index:
        data[i] = data[i]**2
        temp = data[i].sum()
        tap = temp/len(data[i])                                               # time average power
        power_allscale.append(tap)
    for j in range(len(power_allscale)):
        power_allscale[j]=power_allscale[j]/scales[index[j]]
    Etot=np.sqrt(np.sum(power_allscale)/5000/1000)
    return Etot

#=============================================================================#
######### feature extraction: CNN input, dataframe output #####################
#=============================================================================#
def ExtraFeatures(data,fs):
#    emd = EMD(spline_kind='slinear', nbsym=10, extrema_detection='parabol')
    wavelet = 'cmor1.5-1.0'
    scales = np.arange(1, 256)
    df_test = pd.DataFrame()

    for n in range(len(data[:,0,0])):
        a_sample=data[n,:,0]      
        # envelope spectrum domain statistic features
        a_envelope,Freq=envelope_powerspectrum(a_sample,fs,200)
        RMS_envelope=rms(a_envelope)
        SRS_envelope=sra(a_envelope)
        KV_envelope=kv(a_envelope)
        SV_envelope=sv(a_envelope)
        PPV_envelope=ppv(a_envelope)
        CF_envelope=cf(a_envelope)
        IF_envelope=iF(a_envelope)
        MF_envelope=mf(a_envelope)
        SF_envelope=sf(a_envelope)
        KF_envelope=kf(a_envelope)             
        # decomposition features
    #    IMF_EMD = emd.emd(a_sample,t, max_imf=6)
    #    EG2EP_EMD=EG2EP(IMF_EMD)
        # time-frequency spectrum statistic features
        [cfs, frequencies] = pywt.cwt(a_sample, scales, wavelet, 1/fs)
        power = (abs(cfs)) ** 2            
        TFSF_CWT=iFEn_2D(power,frequencies)
        aspower=waveletpower(power,frequencies,scales)
    #    features=[KV_envelope,EG2EP_EMD,TFSF_CWT]
        features=[RMS_envelope,SRS_envelope,KV_envelope,SV_envelope,PPV_envelope,
                      CF_envelope,IF_envelope,MF_envelope,SF_envelope,
                      KF_envelope,TFSF_CWT,aspower]
    #    df_features=pd.DataFrame(features,index=['KurtosisValue_ES','Energy2Entropy_EMD','Energy_CWT'])
        df_features=pd.DataFrame(features,index=['RMS_ES','SquartRoot_ES','KurtosisValue_ES','SkewnessValue_ES','Peak2Peak_ES','CrestFactor_ES','ImpulseFactor_ES',
        'MarginFactor_ES','ShapeFactor_ES','KurtosisFactor_ES','Kurtosis_CWT','Energy_CWT'])  
        df_features=pd.DataFrame.transpose(df_features)
        df_test=df_test.append(df_features)
        print('progress: ' + str(round(100*n/len(data[:,0,0]))) + '%')
    df_output=(df_test-df_test.mean())/df_test.std()
    return df_output