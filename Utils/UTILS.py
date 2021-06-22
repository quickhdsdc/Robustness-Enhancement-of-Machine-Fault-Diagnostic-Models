###############################################################################
######################### Utility Functions ###################################
###############################################################################
import pickle
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.signal import hilbert, detrend
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score

# loads data from one speed range and processes into samples
def load_speedrange(dpath,speed,sample_size,fs,fe=None,classes='all',scaling='Standard',scale=True):
    
    X, y = prepare_samples(dpath+speed,sample_size,fs,fe,classes,scaling,scale)
    
    return X, y


# load and prepcrocess 
def prepare_samples(filepath, sample_size=10000, fs=5000, fe=None, classes='all',scaling='Standard',scale=True):
        
    # different folder structure for Bosch
    if 'BOSCH' in filepath:
        rides_good = read_rides_csv(filepath+'/Good/')
        rides_bad1 = read_rides_csv(filepath+'/Bad/10mm/')
        rides_bad2 = read_rides_csv(filepath+'/Bad/50mm/')
        # rides_bad = rides_bad1 + rides_bad2
        rides_bad = rides_bad2       
    else:
        rides_good = read_rides_csv(filepath+'/Good/')
        rides_bad = read_rides_csv(filepath+'/Bad/')
       
    # only keep acceleration, not speed
    samples_good = rides_to_samples(rides_good)[:,0]
    samples_bad = rides_to_samples(rides_bad)[:,0]

    # interpolate data to desired sampling frequency
    if ('LEILA' in filepath)  or ('Entgleis' in filepath):
        samples_good = interpolate(samples_good, 500, fs)
        samples_bad = interpolate(samples_bad, 500, fs)

    else:
        samples_good = interpolate(samples_good, 5000, fs)
        samples_bad = interpolate(samples_bad, 5000, fs)
    
    # reshape data to desired sample length
    # data is cropped to the biggest value with which reshape is possible
    samples_good = samples_good[:len(samples_good)-len(samples_good)%sample_size].reshape(-1,sample_size)
    samples_bad = samples_bad[:len(samples_bad)-len(samples_bad)%sample_size].reshape(-1,sample_size)
    
    # eliminate samples that are 0
    std_good = np.std(samples_good,axis=1)
    idx_good = np.where(std_good>1e-3)
    samples_good = samples_good[idx_good]
    
    std_bad = np.std(samples_bad,axis=1)
    idx_bad = np.where(std_bad>1e-3)
    samples_bad = samples_bad[idx_bad]
    
    # create labels for data
    labels_good = np.zeros(len(samples_good))
    labels_bad = np.ones(len(samples_bad))
    
    # create sample and label array
    if classes=='all':
        x = np.concatenate((samples_good,samples_bad))
        y = np.concatenate((labels_good,labels_bad))
      
    elif classes=='good':
        x = np.asarray(samples_good)
        y = np.asarray(labels_good)
        
    elif classes=='bad':
        x = np.asarray(samples_bad)
        y = np.asarray(labels_bad)
        
    if fe=='env':
        x = abs(hilbert(x))

    elif 'fft' in fe: # fast-fourier-transform
        
        # only positive frequencies are kept due to symmetry of fft
        x = detrend(x)
        sample_size = f_lim = 512 #  max frequency is 512Hz, frequency sampled at 0.5Hz intervals -> array has length 1024
        x = abs(np.fft.fft(x))[:,1:2*f_lim+1]

    elif 'env_spec' in fe: # envelop spectrum
    
        x = detrend(x)
        sample_size = f_lim = 512 #  max frequency is 512Hz, frequency sampled at 0.5Hz intervals -> array has length 1024
        x = hilbert(x)
        x = abs(np.fft.fft(x))[:,1:2*sample_size+1]
        
    if fe=='None' or fe=='cnn':
    
        scaler = StandardScaler()
        x = scaler.fit_transform(x.reshape(-1,1)).reshape(-1,sample_size)
   
    return x, y

# given the speed v, calculates the time (s) for one 
# rotation of the wheels (d - diameter in m)) 

def time_rotation(v,d=0.92):
    if v>0:
        v = v/3.6
        n = (v/(d/2))/(2*np.pi)
        t = 1/n
    else:
        t = np.inf
    return t

#Function: read_rides
# read all ride samples (.csv) of one speed range

# Inputs:
# filepath - path to folder with samples
# dType - type of data (e.g. 'Bad', 'Good', 'Bad/10mm')

# Returns:
# rides - list of all samples as arrays
        
def read_rides_csv(filepath):                                          
    
    files = os.listdir(filepath)
    rides = []
    if len(files)>0:
        for f in files:
            file = filepath + '/' + f
            try:
                ride = pd.read_csv(file, index_col=False)
                rides.append(ride.to_numpy(copy=True))
            except:
                pass
    else:
        rides=[]
     
    return rides

def read_rides_npy(filepath):                                          
    
    files = os.listdir(filepath)
    rides = []
    if len(files)>0:
        for f in files:
            if f[-4:]=='.npy':
                file = filepath + '/' + f
                ride = np.load(file, allow_pickle=True)
                rides.append(ride)
    else:
        rides=[]
     
    return rides

# concatenates input and returns zeros if input is empty 
def rides_to_samples(rides):
    try:
        samples = np.concatenate((rides))
        return samples
    except:
        return np.zeros(shape=(2,2))


# interpolate sequence with frequency n_in to frequency n_out
def interpolate(sequence, n_in, n_out):
    ratio = n_out/n_in
    x = int(ratio*len(sequence))
    
    seq_new = np.interp(np.arange(0,x), ratio*np.arange(0,len(sequence)), sequence)
    
    return seq_new
     
    

''' 
Function: unify_sample_length
    
Unify the sample length, so that all samples have the same number of data points.
The function calculates the average length of all samples, and pads shorter samples with 0's, 
while longer samples are cropped to the average length.
    
# Input: samples - List of samples
# Output: samples_new - List of samples with uniform length

'''  

def unify_sample_length(samples, method = 'mean'):
    
    length=[]
    for s in samples:
        length.append(len(s))
    
    if method=='mean':
        mean = int(round(np.mean(length)))
        samples_new = []
        for s in samples:
            if len(s)>mean:
                s_new = s[:mean]
                samples_new.append(s_new)
            elif len(s)<mean:
                s_new = s
                s_new.extend([0] * (mean-len(s)))
                samples_new.append(s_new)
            else:
                samples_new.append(s)
        
    elif method=='pad':
        maxLength = np.max(length)
        samples_new = []
        for s in samples:
            if len(s)<maxLength:
                diff = maxLength-len(s)
                s_new = s + [0] * diff
                samples_new.append(s_new)
            else:
                samples_new.append(s)
        
    elif method=='linpol':
        mean = int(round(np.mean(length)))
        x = np.arange(0, mean)
        samples_new = []
        for s in samples:
            if len(s)!=mean:
                s_new = np.interp(x, np.arange(0,len(s)), s)
                samples_new.append(s_new)
            else:
                samples_new.append(s)
                
    samples_new = np.asarray(samples_new)   
        
    return samples_new



'''
Function: train_test_val_split
    
Splits data into training, testing and validation data

Input: 
    X - List of samples
    y - List of labels
    testSize - proportion of testing data (e.g. 0.3)
    valSize - proportin of validation data (e.g. 0.1)
    (trainSize = 1 - testSize - valSize, e.g. trainSize = 1 - 0.3 - 0.1 = 0.6)
    randomState - integer for random shuffling of the data
Output: 
    X_train - List of training samples
    X_test - List of testing samples
    X_val - List of validation samples
    y_train - List of labels for the training samples
    y_test - List of labels for the testing samples
    y_val - List of labels for the validation samples
'''
    
def train_test_val_split(X, y, testSize=0.2, valSize=0.2, randomState=0):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(testSize+valSize), random_state=randomState)

    if valSize>0:
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=(valSize/(testSize+valSize)), random_state=randomState) 
    else:
        X_val, y_val = [], []
        
#    X_train, X_test, X_val, y_train, y_test, y_val = np.asarray(X_train), np.asarray(X_test), np.asarray(X_val), np.asarray(y_train), np.asarray(y_test), np.asarray(y_val)
        
    return X_train, X_test, X_val, y_train, y_test, y_val

# plots TSNE
# features is an array of shape(samples, features)
# labels is an 1d-array of class labels
def plot_tsne(features, labels, perp=50 , title='', save=False ,savepath=None):
        
    labels = labels[:len(features)]
    labels_unique = np.unique(labels)
    lb = []
    for i in range(int(min(labels_unique)), len(labels_unique)+int(min(labels_unique))) :
        lb.append(np.where(labels==i)[0])
    
    tsne = TSNE(n_components=2, perplexity = perp, learning_rate = 200
                ,verbose=2, n_iter=2000, random_state=0, early_exaggeration = 12)
    
    tsne_results = tsne.fit_transform(features)
    tsne2Dx = tsne_results[:,0]
    tsne2Dy = tsne_results[:,1]
    
    fig = plt.figure()

    count=0
    for l in lb:
        c = 'C' + str(count)
        plt.scatter(tsne2Dx[l], tsne2Dy[l], s=2**2, color=c)
        count=count+1
        
    plt.xlabel('tsne-2D-x')
    plt.ylabel('tsne-2D-y')
    plt.title(title)
    
    
    if save==True:
        plt.savefig(savepath)
        plt.close()

    return tsne_results


# plot feature maps 
# TO DO: get model params automatically from model path
# def plot_feature_maps(model_path, x):
#     K.clear_session()
#     directory='./results/' 
    
#     x = x.reshape(-1,1)
    
#     model = create_model('resnetAE', 4096, 32, 0.0001, 'nll', directory+model_path)
#     model.model.load_weights(directory+model_path+'/best_model.hdf5')

#     layer_names = []

#     for layer in model.model.layers:
#         layer_names.append(layer.name) 
    
#     act_index=[position for position, name in enumerate(layer_names) if 'conv' in name or 'pool' in name or 'up' in name or 'bottle' in name or 'z' in name]  # index of activation layers
#     act_index=np.asarray(act_index)

#     actlayer_names=[layer_names[i] for i in act_index]
#     layer_outputs =[model.model.layers[i].output for i in act_index]                      # output of activation layers

    
#     activation_model = models.Model(inputs=model.model.input, outputs=layer_outputs)      # Creates a model that will return these outputs, given the model input
#     test_input = np.expand_dims(x, axis=0)
#     activations = activation_model.predict(test_input)                              # Returns a list of Numpy arrays: one array per layer activation

#     plt.figure()
#     plt.plot(x.reshape(-1),color='#0021D7')
# #    plt.savefig(directory+'convAE/256-500-0.0001-env-8-/')


#     col=[]
#     row=[]
    
#     for layer_name,layer_activation in zip(actlayer_names,activations):
#         i=0
#         n_features = layer_activation.shape[-1] # Number of features in the feature map
#         if layer_activation.ndim>2:
#             size = layer_activation.shape[1]*layer_activation.shape[2] #The feature map has shape (1, size, n_features).
#         else:
#             size = 1
#         images_per_row = int(np.ceil(math.sqrt(n_features)))
#         n_rows = math.ceil(n_features/images_per_row) # Tiles the activation channels in this matrix
#         fig,axes=plt.subplots(nrows=n_rows, ncols=images_per_row)
#         fig.suptitle(layer_name)
#         if n_rows>1:
#             for row in range(n_rows):
#                 for col in range(images_per_row):
#                     if row*images_per_row+col < n_features:
#                         print(n_rows*i+col)
#                         if size>1:
#                             axes[row,col].plot(layer_activation[:,:,n_rows*i+col].reshape(-1),color='#0021D7')
#                             axes[row,col].yaxis.set_visible(False)
#                             axes[row,col].xaxis.set_visible(False)
#                         else:
#                             axes[row,col].set_ylim([0,1])
#                             axes[row,col].vlines(0,0,layer_activation[:,n_rows*i+col].reshape(-1))
#                             axes[row,col].yaxis.set_visible(True)
#                             axes[row,col].xaxis.set_visible(False)
#                     else:
#                         axes[row,col].yaxis.set_visible(False)
#                         axes[row,col].xaxis.set_visible(False)  
#                 i=i+1
#         elif n_rows*images_per_row==1:
#             axes.plot(layer_activation[:,:,0].reshape(-1),color='red')
#         else:
#             for col in range(images_per_row):
                
#                 if col <= 16:
#                     axes[col].plot(layer_activation[:,:,col],color='#0021D7')
#                     axes[col].yaxis.set_visible(False)
#                     axes[col].xaxis.set_visible(False)
#                 else:
#                     axes[col].yaxis.set_visible(False)
#                     axes[col].xaxis.set_visible(False) 
                    
#     code_model = models.Model(inputs=model.enc.input, outputs=model.enc.output)
#     code, shortcut = code_model.predict(test_input)  
#     plt.figure()
#     plt.plot(code.reshape(-1),color='red')
    
                
def create_directory(directory_path):
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile !:(
            return None
        return directory_path

# wrapper function for creating a model
# def create_model(name, input_shape, lr, output_directory,epochs=10000,x_val=None,y_val=None, arch=[16, 'act'], batch_size=64, activation='leakyrelu', conf_inc=0.9, conf_max=10) :
#     if name == 'CAE':  # convolutional autoencoder with residual connections (better)
#         from models.CAE import CAE
#         return CAE(output_directory, input_shape, lr)
#     if name == 'CCAE':  # convolutional autoencoder with residual connections (better)
#         from models.CCAE import CCAE
#         return CCAE(output_directory, input_shape, lr)
#     if name == 'CC':  # convolutional autoencoder with residual connections (better)
#         from models.CC import CC
#         return CC(output_directory, input_shape, lr)
#     if name == 'Baseline_ddcn':  # convolutional autoencoder with residual connections (better)
#         from ddan.ddcn import DDCNModel
#         #opt =  tf.train.AdamOptimizer(1e-3)
#         opt = tf.train.MomentumOptimizer(lr, 0.9)
#         conf_incr =  (conf_max - 1.5)/ (conf_inc*epochs)
#         return DDCNModel(nfeatures=input_shape, arch=arch, mmd_layer_idx=[1], val_data=(x_val, y_val), 
#                          epochs=epochs, batch_size=batch_size, validate_every=100, confusion_incr=conf_incr, confusion=1.5, confusion_max=conf_max,
#                          optimizer=opt, activations=activation)
#     if name == 'Baseline_deepcoral':  # convolutional autoencoder with residual connections (better)
#         from ddan.deepcoral import DeepCoralNet
#         #opt =  tf.train.AdamOptimizer(1e-3)
#         opt = tf.train.MomentumOptimizer(lr, 0.9)
#         conf_incr =  (conf_max - 1.5)/ (conf_inc*epochs)
#         return DeepCoralNet(nfeatures=input_shape, arch=arch, coral_layer_idx=[1], val_data=(x_val, y_val), 
#                          epochs=epochs, batch_size=batch_size, validate_every=100, confusion_incr=conf_incr, confusion=1.5, confusion_max=conf_max,
#                          optimizer=opt, activations=activation)
#     if name=='Baseline_dann':
#         from ddan.dann import DANNModel
#         opt = tf.train.MomentumOptimizer(lr, 0.9)
#         model = DANNModel(nfeatures=input_shape, arch_shared=arch, arch_domain=[4, 'act'], 
#                          arch_clf=[4, 'act'], val_data=(x_val, y_val), epochs=epochs, batch_size=batch_size, validate_every=100,
#                          optimizer=opt, activations=activation)
#         return model
    
def calculate_metrics(y_true, y_pred, duration, y_true_val=None, y_pred_val=None):
    res = pd.DataFrame(data=np.zeros((1, 4), dtype=np.float), index=[0],
                       columns=['precision', 'accuracy', 'recall', 'duration'])
    res['precision'] = precision_score(y_true, y_pred, average='macro')
    res['accuracy'] = accuracy_score(y_true, y_pred)

    if not y_true_val is None:
        # this is useful when transfer learning is used with cross validation
        res['accuracy_val'] = accuracy_score(y_true_val, y_pred_val)

    res['recall'] = recall_score(y_true, y_pred, average='macro')
    res['duration'] = duration
    return res
    
def plot_epochs_metric(hist, file_name, metric='loss'):
    plt.figure()
    plt.plot(hist.history[metric])
    plt.plot(hist.history['val_' + metric])
    plt.title('model ' + metric)
    plt.ylabel(metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()

    
# def save_logs(output_directory, hist, lr=False, y_true_val=None, y_pred_val=None):
#     hist_df = pd.DataFrame(hist.history)
#     hist_df.to_csv(output_directory + 'history.csv', index=False)

#     index_best_model = hist_df['loss'].idxmin()
#     row_best_model = hist_df.loc[index_best_model]

#     df_best_model = pd.DataFrame(data=np.zeros((1, 6), dtype=np.float), index=[0],
#                                  columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_mse',
#                                           'best_model_val_mse', 'best_model_learning_rate', 'best_model_nb_epoch'])

#     df_best_model['best_model_train_loss'] = row_best_model['loss']
#     df_best_model['best_model_val_loss'] = row_best_model['val_loss']
# #    df_best_model['best_model_train_mse'] = row_best_model['mean_squared_error']
# #    df_best_model['best_model_val_mse'] = row_best_model['val_mean_squared_error']
#     if lr == True:
#         df_best_model['best_model_learning_rate'] = row_best_model['lr']
#     df_best_model['best_model_nb_epoch'] = index_best_model

#     df_best_model.to_csv(output_directory + 'df_best_model.csv', index=False)


#     # plot losses
#     if os.path.exists(output_directory + 'epochs_loss.png'):
#         plot_epochs_metric(hist, output_directory + 'epochs_loss_classifier.png')
#     else:
#         plot_epochs_metric(hist, output_directory + 'epochs_loss.png')

#     return 