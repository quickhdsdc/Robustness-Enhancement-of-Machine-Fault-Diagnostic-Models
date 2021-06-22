import numpy as np
import random
from scipy.spatial.distance import euclidean
import operator
from scipy import signal
from Utils.FeatureExtraction import ExtraFeatures
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler()

def augment_train_set_ft(x_train, x_train_ft, N):
    """
    :param x_train: The original train set
    :param x_train_ft: The feature  set 
    :param N: The number of synthetic time series. 
    """
    # synthetic train set and labels 
    synthetic_x_train = []
    fs=5000
    n_sample = np.shape(x_train)[0]    
    x_train = np.reshape(x_train,(n_sample,int(2.048*fs)))
    
    nb_prototypes_per_class = N
    # get the pairwise matrix 
    dist_pair_mat = calculate_dist_matrix_ft(scaler.fit_transform(x_train_ft))
    medoid_idx = medoid_ft(x_train_ft, dist_pair_mat)
    # loop through the number of synthtectic examples needed
    for n in range(nb_prototypes_per_class): 
        print('n='+str(n))
        # get the weights and the init for avg method 
        x_train_subsets, x_train_ft_subsets, weights_subset, init_avg = get_weights_average_selected(x_train, x_train_ft, medoid_idx, dist_pair_mat)
        # get the synthetic data 
        synthetic_mts = dba(x_train_subsets, x_train_ft_subsets, fs, max_iter = 10, verbose=True, weights=weights_subset, init_avg_series = init_avg)  
        # add the synthetic data to the synthetic train set
        synthetic_x_train.append(synthetic_mts)
    # return the synthetic set 
    return np.array(synthetic_x_train)


def dba(tseries, ftvector, fs, max_iter =10, verbose=True, init_avg_series = None, weights=None): 

    # check if given dataset is empty 
    if len(tseries)==0: 
        # then return a random time series because the average cannot be computed 
        start_idx = np.random.randint(0,len(tseries))
        return np.copy(tseries[start_idx])
    
    avg = init_avg_series
        
    if len(tseries) == 1:
        return avg
    if verbose == True: 
        print('Doing iteration')
        
    # main DBA loop 
    for i in range(max_iter):
        if verbose == True:
            print(' ',i,'...')
        if weights is None:
            # when giving all time series a weight equal to one we have the 
            # non - weighted version of DBA 
            weights = np.ones((len(tseries),tseries.shape[1]), dtype=np.float64)
        # dba iteration 
        avg = _dba_iteration(tseries, ftvector, avg, weights, fs)    
    return avg 


def _dba_iteration(tseries, ftvector, avg, weights, fs):
    """
    Perform one weighted dba iteration and return the new average 
    """
    # the number of time series in the set
    n = len(tseries)
    # length of the time series 
    ntime = avg.shape[0]
    # features of avg
    avg_ft = ExtraFeatures(avg,fs)
    scaler.fit(ftvector)    
    # number of dimensions (useful for MTS)
    # array containing the new weighted average sequence 
    new_avg = np.zeros((ntime),dtype=np.float64) 
    # array of sum of weights 
    sum_weights = np.zeros((ntime),dtype=np.float64)
    # loop the time series 
    for s in range(n): 
        series = tseries[s]
        ft = ftvector[s]
        dist = np.linalg.norm(scaler.transform(avg_ft.reshape(1,-1))-scaler.transform(ft.reshape(1,-1)))
        new_avg = new_avg + dist*weights[s]*series
        sum_weights = sum_weights + dist*weights[s]
        
    new_avg = new_avg/sum_weights
    
    return new_avg


def get_weights_average_selected(x_train, x_train_ft, medoid_idx, dist_pair_mat):
    # get the number of dimenions 
    num_dim = x_train.shape[1]
    num_sample = x_train.shape[0]
    # maximum number of sub set samples 
    n = 40
    # sub k nearst neighbors
    subk = 10
    # the weight for the center 
    weight_center = 0.5 
    # the total weight of the neighbors
    weight_neighbors = 0.3
    # total weight of the non neighbors 
    weight_remaining = 1.0- weight_center - weight_neighbors
    # number of non neighbors 
    n_others = n - 1 - subk
    # get the weight for each non neighbor 
    if n_others == 0 : 
        fill_value = 0.0
    else:
        fill_value = weight_remaining/n_others
    # choose a random time series 
    # idx_center = random.randint(0,num_sample-1)
    idx_center = medoid_idx
    # get the init dba 
    init_dba = x_train[idx_center]
    # init the weight matrix or vector for univariate time series 
    weights = np.full((num_sample,num_dim),fill_value,dtype=np.float64)
    # fill the weight of the center 
    weights[idx_center] = weight_center
    # find the top k nearest neighbors
    subset_idx = np.array(get_neighbors(x_train,init_dba,n,pre_computed_matrix=dist_pair_mat,
                         index_test_instance= idx_center))
    # select a subset of the k nearest neighbors 
    final_neighbors_idx = np.random.permutation(n)[:subk]
    # adjust the weight of the selected neighbors 
    weights[subset_idx[final_neighbors_idx]] = weight_neighbors / subk
    # return the weights and the instance with maximum weight (to be used as 
    # init for DBA )
    weights_subset = weights[subset_idx]
    x_train_subsets = x_train[subset_idx]
    x_train_ft_subsets = x_train_ft[subset_idx]
    return x_train_subsets, x_train_ft_subsets, weights_subset, init_dba


def get_neighbors(x_train, x_test_instance, k, pre_computed_matrix=None, 
                  index_test_instance=None, return_distances = False): 
    """
    Given a test instance, this function returns its neighbors present in x_train
    NB: If k==0 zero it only returns the distances
    """
    distances = []
    # loop through the training set 
    for i in range(len(x_train)): 
        # calculate the distance between the test instance and each training instance
        dist = pre_computed_matrix[i,index_test_instance]
        # add the index of the current training instance and its corresponding distance 
        distances.append((i, dist))
    # if k (nb_neighbors) is zero return all the items with their distances 
    # NOT SORTED 
    if k==0: 
        if return_distances == True: 
            return distances
        else:
            print('Not implemented yet')
            exit()
    # sort list by specifying the second item to be sorted on 
    distances.sort(key=operator.itemgetter(1))
    # else do return only the k nearest neighbors
    neighbors = []
    for i in range(k): 
        if return_distances == True: 
            # add the index and the distance of the k nearest instances from the train set 
            neighbors.append(distances[i])
        else:
            # add only the index of the k nearest instances from the train set 
            neighbors.append(distances[i][0])        
    return neighbors

def calculate_dist_matrix_ft(ftvectors):
    N = len(ftvectors)
    pairwise_dist_matrix = np.zeros((N,N), dtype = np.float64)
    # pre-compute the pairwise distance
    for i in range(N-1):
        x = ftvectors[i]
        for j in range(i+1,N):
            y = ftvectors[j] 
            dist = np.linalg.norm(x-y)           
            pairwise_dist_matrix[i,j] = dist 
            # dtw is symmetric 
            pairwise_dist_matrix[j,i] = dist 
        pairwise_dist_matrix[i,i] = 0 
    return pairwise_dist_matrix

def medoid_ft(ftvectors,pairwise_dist_matrix):
    """
    Calculates the medoid of the given list of MTS
    :param tseries: The list of time series 
    """
    N = len(ftvectors)
    if N == 1 : 
        return 0  
    # pairwise_dist_matrix = calculate_dist_matrix_ft(ftvectors)     
    sum_dist = np.sum(pairwise_dist_matrix, axis = 0)
    min_idx = np.argmin(sum_dist)
    return min_idx

#=============================================================================#
############################ read feature sets ###############################
#=============================================================================# 

def readdata(speeds,health,dataset,baseline):   
    if baseline == 'ResNet':
        if health == 'Good':
            varname_good = 'a_ESZUEG_Good_' + speeds[0] + '_' + dataset[0]
            x = np.load('./Data/Axlebox_trackirr/' + varname_good + '.npy')         
        else:
            varname_bad = 'a_ESZUEG_Bad_' + speeds[0] + '_' + dataset[0]                        
            x = np.load('./Data/Axlebox_trackirr/' + varname_bad + '.npy')          
    elif baseline == 'CWT+ResNet':
        if health == 'Good':            
            varname_good = 'a_ESZUEG_Good_' + speeds[0] + '_' + dataset[0] + '_sacwt'
            x = np.load('./Data/All_featurespace/' + varname_good + '.npy')           
        else:
            varname_bad = 'a_ESZUEG_Bad_' + speeds[0] + '_' + dataset[0]  + '_sacwt'         
            x = np.load('./Data/All_featurespace/' + varname_bad + '.npy')               
    elif baseline == 'HT+ResNet':
        if health == 'Good':
            varname_good = 'a_ESZUEG_Good_' + speeds[0] + '_' + dataset[0] + '_env'
            x = np.load('./Data/All_featurespace/' + varname_good + '.npy')          
        else:
            varname_bad = 'a_ESZUEG_Bad_' + speeds[0] + '_' + dataset[0]  + '_env'                     
            x = np.load('./Data/All_featurespace/' + varname_bad + '.npy')           
    elif baseline == 'GBDT':
        if health == 'Good':
            varname_good = 'a_ESZUEG_Good_' + speeds[0] + '_' + dataset[0] + '_sf'
            x = np.load('./Data/All_featurespace/' + varname_good + '.npy')    
        else:
            varname_bad = 'a_ESZUEG_Bad_' + speeds[0] + '_' + dataset[0]  + '_sf'                     
            x = np.load('./Data/All_featurespace/' + varname_bad + '.npy')                     
        
    n = len(x)
    if health == 'Good':    
        y=np.hstack((np.reshape(np.zeros(n),(n,1)),np.reshape(np.ones(n),(n,1)))) # 0 1 for good    
    else:
        y=np.hstack((np.reshape(np.ones(n),(n,1)),np.reshape(np.zeros(n),(n,1)))) # 1 0 for bad    
    
    if baseline == 'GBDT':
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
    else:
        scaler = MinMaxScaler()
        x = scaler.fit_transform(x.T).T    
        x = x.reshape((x.shape[0],x.shape[1],1))
        
    indices = list(range(len(x)))  
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]        
    
    return x,y   

#=============================================================================#
############################### miscellanea ###################################
#=============================================================================#            
def segment_signal(data,col,window_size = 10000):
    a = np.empty((0,window_size))
    for index_start in range(0,data[col].count()-window_size,window_size//40):
        for (start, end) in windows(data[col], window_size,0.5,index_start):
            segment = data[col][start:end]
            if(len(data[col][start:end]) == window_size):
                a = np.vstack([a,segment])
    return a   

def windows(data, size, overlap,start):
    while start < data.count():
        yield int(start), int(start + size)
        start += (size*overlap)


def SNR_Noise_2D(signal,SNR_db):
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

def SNR_Noise_1D(signal,SNR_db):
    Npts = np.shape(signal)[0]
    signal_noise=np.zeros((Npts))
    sig=signal
    noise = np.random.normal(0,1,Npts)
    sig_power = np.mean(sig**2)
    noise_power=np.mean(noise**2)
    K=(sig_power/noise_power)*10**(-SNR_db/10)
    new_noise=np.sqrt(K)*noise
    sig = sig + new_noise
    signal_noise= sig       
    return signal_noise 


# synthetic_x_train = augment_train_set(x_train, x_train_features, 100)
# pairwise_dist_matrix = calculate_dist_matrix_ft(x_train_features)
# medoid_idx = medoid_ft(x_train_features)
# x_train_subsets, weights_subset, init_dba = get_weights_average_selected(x_train, medoid_idx, pairwise_dist_matrix)