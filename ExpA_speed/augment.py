import numpy as np
import random
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import operator
from scipy import signal

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
        if pre_computed_matrix is None: 
#            dist , _ = fastdtw(x_test_instance, x_train[i,:],len(x_test_instance)//2,euclidean)
            dist , _ = fastdtw(x_test_instance, x_train[i,:],10,euclidean)           
        else: 
            # do not re-compute the distance just get it from the precomputed one
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

def get_weights_average_selected(c_x_train, dist_pair_mat):
    # get the number of dimenions 
    num_dim = c_x_train.shape[1]
    num_sample = c_x_train.shape[0]
    # maximum number of sub set samples 
    n = 10
    # maximum number of K for KNN 
    max_k = 5 
    # maximum number of sub neighbors 
    max_subk = 2
    # get the real k for knn 
    k = 5
    # make sure 
    subk = min(max_subk,k)
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
    idx_center = random.randint(0,num_sample-1)
    # get the init dba 
    init_dba = c_x_train[idx_center]
    # init the weight matrix or vector for univariate time series 
    weights = np.full((num_sample,num_dim),fill_value,dtype=np.float64)
    # fill the weight of the center 
    weights[idx_center] = weight_center
    # find the top k nearest neighbors
    subset_idx = np.array(get_neighbors(c_x_train,init_dba,n,pre_computed_matrix=dist_pair_mat,
                         index_test_instance= idx_center))
    # select a subset of the k nearest neighbors 
    final_neighbors_idx = np.random.permutation(n)[:subk]
    # adjust the weight of the selected neighbors 
    weights[subset_idx[final_neighbors_idx]] = weight_neighbors / subk
    # return the weights and the instance with maximum weight (to be used as 
    # init for DBA )
    weights_subset = weights[subset_idx]
    c_x_train_subsets = c_x_train[subset_idx]
    return c_x_train_subsets, weights_subset, init_dba

def calculate_dist_matrix(tseries,fs):
    N = len(tseries)
    pairwise_dist_matrix = np.zeros((N,N), dtype = np.float64)
    # pre-compute the pairwise distance
    for i in range(N-1):
        x = tseries[i]
        for j in range(i+1,N):
            y = tseries[j] 
#            dist,_ = fastdtw(x,y,len(y)//2)
            dist,_ = fastdtw(x,y,fs//10)           
            # because dtw returns the sqrt
#            dist = dist*dist 
            pairwise_dist_matrix[i,j] = dist 
            # dtw is symmetric 
            pairwise_dist_matrix[j,i] = dist 
        pairwise_dist_matrix[i,i] = 0 
    return pairwise_dist_matrix

def medoid(tseries):
    """
    Calculates the medoid of the given list of MTS
    :param tseries: The list of time series 
    """
    N = len(tseries)
    if N == 1 : 
        return 0,tseries[0]
    pairwise_dist_matrix = calculate_dist_matrix(tseries)
        
    sum_dist = np.sum(pairwise_dist_matrix, axis = 0)
    min_idx = np.argmin(sum_dist)
    med = tseries[min_idx]
    return min_idx, med

def _dba_iteration(tseries, avg,weights):
    """
    Perform one weighted dba iteration and return the new average 
    """
    # the number of time series in the set
    n = len(tseries)
    # length of the time series 
    ntime = avg.shape[0]
    # number of dimensions (useful for MTS)
    # array containing the new weighted average sequence 
    new_avg = np.zeros((ntime),dtype=np.float64) 
    # array of sum of weights 
    sum_weights = np.zeros((ntime),dtype=np.float64)
    # loop the time series 
    for s in range(n): 
        series = tseries[s]
        dtw_dist, dtw = fastdtw(avg, series,50)
        new_avg = new_avg + dtw_dist*weights[s]*series
        sum_weights = sum_weights + dtw_dist*weights[s]
        
    new_avg = new_avg/sum_weights
    
    return new_avg
        
def dba(tseries, max_iter =10, verbose=False, init_avg_method = 'medoid', 
        init_avg_series = None, weights=None): 
    """
    Computes the Dynamic Time Warping (DTW) Barycenter Averaging (DBA) of a 
    group of Multivariate Time Series (MTS). 
    :param tseries: A list containing the series to be averaged, where each 
        MTS has a shape (l,m) where l is the length of the time series and 
        m is the number of dimensions of the MTS - in the case of univariate 
        time series m should be equal to one
    :param max_iter: The maximum number of iterations for the DBA algorithm.
    :param verbose: If true, then provide helpful output.
    :param init_avg_method: Either: 
        'random' the average will be initialized by a random time series, 
        'medoid'(default) the average will be initialized by the medoid of tseries, 
        'manual' the value in init_avg_series will be used to initialize the average
    :param init_avg_series: this will be taken as average initialization if 
        init_avg_method is set to 'manual'
    :param distance_algorithm: Determine which distance to use when aligning 
        the time series
    :param weights: An array containing the weights to calculate a weighted dba
        (NB: for MTS each dimension should have its own set of weights)
        expected shape is (n,m) where n is the number of time series in tseries 
        and m is the number of dimensions
    """
    # check if given dataset is empty 
    if len(tseries)==0: 
        # then return a random time series because the average cannot be computed 
        start_idx = np.random.randint(0,len(tseries))
        return np.copy(tseries[start_idx])
    
    # init DBA
    if init_avg_method == 'medoid':
        avg = np.copy(medoid(tseries)[1])
    elif init_avg_method == 'random': 
        start_idx = np.random.randint(0,len(tseries))
        avg = np.copy(tseries[start_idx])
    else: # init with the given init_avg_series
        avg = np.copy(init_avg_series)
        
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
        avg = _dba_iteration(tseries,avg,weights)
    
    return avg 


def augment_train_set(x_train, y_train, N):
    """
    This method takes a dataset and augments it using the method in icdm2017. 
    :param x_train: The original train set
    :param y_train: The original labels set 
    :param N: The number of synthetic time series. 
    :param dba_iters: The number of dba iterations to converge.
    :param weights_method_name: The method for assigning weights (see constants.py)
    :param distance_algorithm: The name of the distance algorithm used (see constants.py)
    """
    # synthetic train set and labels 
    synthetic_x_train = []
    synthetic_y_train = []
    fs=5000
    n_sample = np.shape(x_train)[0]
    c_x_train = np.zeros((n_sample,fs*2//10))
    
    x_train = np.reshape(x_train,(n_sample,2*fs))
    
    # loop through each class
    for c in range(1): 
        # get the MTS for this class 
        x_train = x_train[np.where(y_train[:,0]==c)]
        x_train = x_train[:,:]
        len(x_train)
        
        [b,a]=signal.butter(5,[200/(5000/2)],'low')
        for i in range(len(x_train)):
            each=signal.filtfilt(b, a, x_train[i,:])
            c_x_train[i,:]=signal.resample(each,1000)

        nb_prototypes_per_class = N
        # get the pairwise matrix 
        dist_pair_mat = calculate_dist_matrix(c_x_train,fs//10)
        # loop through the number of synthtectic examples needed
        for n in range(nb_prototypes_per_class): 
            print('n='+str(n))
            # get the weights and the init for avg method 
            c_x_train_subsets, weights_subset, init_avg = get_weights_average_selected(c_x_train,dist_pair_mat)
            # get the synthetic data 
            synthetic_mts = dba(c_x_train_subsets, 1, verbose=True, 
                            weights=weights_subset,
                            init_avg_method = 'manual',
                            init_avg_series = init_avg)  
            # add the synthetic data to the synthetic train set
            synthetic_x_train.append(synthetic_mts)
            # add the corresponding label 
            synthetic_y_train.append(c)
    # return the synthetic set 
    return np.array(synthetic_x_train), np.array(synthetic_y_train)

#x_syn_peak, y_syn_peak=augment_train_set(xraw_test_peak[0:49,:], y_test_peak[0:49,:],100)
#x_syn_per, y_syn_per=augment_train_set(xraw_test_per, y_test_per,100)
x_syn_osc, y_syn_osc=augment_train_set(xraw_test_osc, y_test_osc,100)


