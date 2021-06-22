# compute distances between domains
import numpy as np
from Utils.mmd import compute_mmd_on_samples
import os

def compute_domain_distance(domain1, domain2, n=500):
    """
    ------------------------------------------------------------------
    Compute the domain distance between two domains based on the mean feature vetors
    of both respective domains. Features are extracted based on the specified feature
    extraction method.
    ------------------------------------------------------------------
    Args:
        domain1 : array, shape (samples,features)
        domain2 : array, shape (samples,features)
        n : int, number of random samples used to compute the mean feature vectors

    ------------------------------------------------------------------
    Returns:
        mmd : Maximum Mean Discrepancy between domains
    ------------------------------------------------------------------
    """    
    ### select 500 random samples from each domain

    # get index for n random samples
    index_domain1 = np.random.choice(len(domain1), n, replace=False)
    index_domain2 = np.random.choice(len(domain2), n, replace=False)
    
    print('\nDomain 1 : {} , Domain 2: {}'.format(domain1.shape,domain2.shape))
    print('{} samples chosen randomly from each domain'.format(n))
    print(50*'-')
    
    samples_domain1 = domain1[index_domain1]
    samples_domain2 = domain2[index_domain2]

    mmd = compute_mmd_on_samples(samples_domain1, samples_domain2)[0]
    
    print('Maximum Mean Discrepancy: ', mmd)

    return mmd

dataset = ['ESZUEG','BOSCH','LEILA','Entgleis']
speeds = ['v15v25','v25v35','v35v45','v45v55','v55v65','v65v75','v75v85','v85v95', 'v95v105']
health = ['Bad','Good']
feature_space = ['fft','env','sacwt','sf']

sample_size=10240
fs = 5000

# read all feature data
folderpath = './Data/All_featurespace/'
for file in os.listdir(folderpath):
    if file.endswith('.npy'):
        varname, ext = os.path.splitext(file)
        filepath = os.path.join(folderpath, file)
        globals()[varname] = np.load(filepath)

# domain distance between different speed ranges
for j in range(len(dataset)):
    if dataset[j] == 'BOSCH':
        speeds = ['v15v25','v25v35','v35v45']   
    elif dataset[j] == 'LEILA':
        speeds = ['v15v25','v25v35','v35v45','v45v55','v55v65','v65v75','v75v85']
    elif dataset[j] == 'Entgleis':
        speeds = ['v0v15','v15v25','v25v35','v35v45','v45v55']   
    elif dataset[j] == 'ESZUEG':
        speeds = ['v15v25','v25v35','v35v45','v45v55','v55v65','v65v75','v75v85','v85v95', 'v95v105']    
    for k in range(len(health)):
        for m in range(len(feature_space)):
            varname_domain_ref = 'a_' + dataset[j] + '_' + health[k] + '_' + speeds[0] + '_' + feature_space[m]
            domain_ref = globals()[varname_domain_ref]   
            varname_mmd_allspeeds = 'mmd_' + dataset[j] + '_' + health[k] + '_' + feature_space[m] + '_' + speeds[0] + '_all'
            globals()[varname_mmd_allspeeds] = []
            for i in range(1,len(speeds)):
                varname_domain2 = 'a_' + dataset[j] + '_' + health[k] + '_' + speeds[i] + '_' + feature_space[m]
                domain2 = globals()[varname_domain2]
                samples = min(len(domain_ref),len(domain2),300)
                varname_mmd = 'mmd_' + dataset[j] + '_' + health[k] + '_' + feature_space[m] + '_' + speeds[0] + '_' + speeds[i]                                
                globals()[varname_mmd] = compute_domain_distance(domain_ref, domain2, n=samples)
                globals()[varname_mmd_allspeeds].append(globals()[varname_mmd])
            globals()[varname_mmd_allspeeds] = np.asarray(globals()[varname_mmd_allspeeds])

                
# domain distance between different datasets
speeds = ['v15v25','v25v35','v35v45']
for k in range(len(health)):
    for m in range(len(feature_space)):
        for i in range(len(speeds)):
            varname_domain_ref = 'a_' + dataset[0] + '_' + health[k] + '_' + speeds[i] + '_' + feature_space[m] 
            domain_ref = globals()[varname_domain_ref]   
            varname_mmd_alldatasets = 'mmd_' + health[k] + '_' + speeds[i] + '_' + feature_space[m] + '_' + dataset[0] + '_all'
            globals()[varname_mmd_alldatasets] = []                     
            for j in range(1,len(dataset)):
                varname_domain2 = 'a_' + dataset[j] + '_' + health[k] + '_' + speeds[i] + '_' + feature_space[m]
                domain2 = globals()[varname_domain2]
                samples = min(len(domain_ref),len(domain2),300)
                varname_mmd = 'mmd_' + health[k] + '_' + speeds[i] + '_' + feature_space[m] + '_' + dataset[0] + '_' + dataset[j]
                globals()[varname_mmd] = compute_domain_distance(domain_ref, domain2, n=samples)
                globals()[varname_mmd_alldatasets].append(globals()[varname_mmd])
            globals()[varname_mmd_alldatasets] = np.asarray(globals()[varname_mmd_alldatasets])
       
            
# domain distance between clean and corrupted datasets
domain_ref = a_ESZUEG_Bad_v35v75_clean_env
domain2 = a_ESZUEG_Bad_v35v75_WFbad_env
samples = min(len(domain_ref),len(domain2),300)
mmd_ESZUEG_Bad_v35v75_WFbad_env = compute_domain_distance(domain_ref, domain2, n=samples)

domain_ref = a_ESZUEG_Good_v35v75_clean_env
domain2 = a_ESZUEG_Good_v35v75_osc_env
samples = min(len(domain_ref),len(domain2),300)
mmd_ESZUEG_Good_v35v75_osc_env = compute_domain_distance(domain_ref, domain2, n=samples)

domain_ref = a_ESZUEG_Good_v35v75_clean_env
domain2 = a_ESZUEG_Good_v35v75_peak_env
samples = min(len(domain_ref),len(domain2),300)
mmd_ESZUEG_Good_v35v75_peak_env = compute_domain_distance(domain_ref, domain2, n=samples)

domain_ref = a_ESZUEG_Good_v35v75_clean_env
domain2 = a_ESZUEG_Good_v35v75_per_env
samples = min(len(domain_ref),len(domain2),300)
mmd_ESZUEG_Good_v35v75_per_env = compute_domain_distance(domain_ref, domain2, n=samples)

mmd_ESZUEG_trackirr = np.asarray([mmd_ESZUEG_Bad_v35v75_WFbad_env, mmd_ESZUEG_Good_v35v75_osc_env, 
                                  mmd_ESZUEG_Good_v35v75_peak_env, mmd_ESZUEG_Good_v35v75_per_env])

# distance between two classes within source domain

varname_mmd_allspeeds = 'mmd_ESZUEG_Good_Bad_env_all'
globals()[varname_mmd_allspeeds] = []
for i in range(len(speeds)):
    varname_domain_ref = 'a_ESZUEG_Good_' + speeds[i] + '_env'
    domain_ref = globals()[varname_domain_ref]       
    varname_domain2 = 'a_ESZUEG_Bad_' + speeds[i] + '_env'
    domain2 = globals()[varname_domain2]
    samples = min(len(domain_ref),len(domain2),300)
    varname_mmd = 'mmd_ESZUEG_Good_Bad_env_' +  speeds[i]                               
    globals()[varname_mmd] = compute_domain_distance(domain_ref, domain2, n=samples)
    globals()[varname_mmd_allspeeds].append(globals()[varname_mmd])
globals()[varname_mmd_allspeeds] = np.asarray(globals()[varname_mmd_allspeeds])
