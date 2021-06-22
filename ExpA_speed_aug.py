import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from Models.ResNet import test_CNN, test_GBDT, ResNet, ResNet_env
import tensorflow as tf
from scipy.signal import resample

def data4balance(speeds,dataset,baseline):
    good=[]
    bad=[]
    good_rest=[]
    for i in range(len(speeds)):
        if baseline == 'ResNet':
            varname_good = 'a_' + dataset[0] + '_Good_' + speeds[i]
            varname_bad = 'a_' + dataset[0] + '_Bad_' + speeds[i]            
            good_speed = np.load('./Data/All_speeds/' + varname_good + '.npy')
            bad_speed = np.load('./Data/All_speeds/' + varname_bad + '.npy')   
        elif baseline == 'CWT+ResNet':
            varname_good = 'a_' + dataset[0] + '_Good_' + speeds[i] + '_sacwt'
            varname_bad = 'a_' + dataset[0] + '_Bad_' + speeds[i]  + '_sacwt'         
            good_speed = np.load('./Data/All_featurespace/' + varname_good + '.npy')
            bad_speed = np.load('./Data/All_featurespace/' + varname_bad + '.npy')               
        elif baseline == 'HT+ResNet':
            varname_good = 'a_' + dataset[0] + '_Good_' + speeds[i] + '_env'
            varname_bad = 'a_' + dataset[0] + '_Bad_' + speeds[i]  + '_env'         
            good_speed = np.load('./Data/All_featurespace/' + varname_good + '.npy')
            bad_speed = np.load('./Data/All_featurespace/' + varname_bad + '.npy')   
        elif baseline == 'GBDT':
            varname_good = 'a_' + dataset[0] + '_Good_' + speeds[i] + '_sf'
            varname_bad = 'a_' + dataset[0] + '_Bad_' + speeds[i]  + '_sf'         
            good_speed = np.load('./Data/All_featurespace/' + varname_good + '.npy')
            bad_speed = np.load('./Data/All_featurespace/' + varname_bad + '.npy')                   
        n_good = len(good_speed)        
        n_bad = len(bad_speed)        
        
        if n_good>n_bad:    
            indices = list(range(n_good))   
            np.random.shuffle(indices)
            idx_good = indices[:n_bad]
            idx_rest = indices[n_bad:]
            good_speed_rest = good_speed[idx_rest]
            good_speed = good_speed[idx_good] 
            good_rest.append(good_speed_rest)
        else:
            indices = list(range(n_bad))   
            np.random.shuffle(indices)
            idx_bad = indices[:n_good]
            bad_speed = bad_speed[idx_bad]
        
        if len(good_speed)>400:
            good.append(good_speed[:400])
            bad.append(bad_speed[:400])
        else:
            good.append(good_speed)
            bad.append(bad_speed)        
        
    good=np.vstack(good)
    n = len(good)
    label_good=np.hstack((np.reshape(np.zeros(n),(n,1)),np.reshape(np.ones(n),(n,1)))) # 0 1 for good
    bad=np.vstack(bad)
    n = len(bad)
    label_bad=np.hstack((np.reshape(np.ones(n),(n,1)),np.reshape(np.zeros(n),(n,1))))  # 1 0 for bad  
    if good_rest!=[]:
        good_rest=np.vstack(good_rest)
    
    x = np.vstack((good,bad))
    y = np.vstack((label_good,label_bad))
    if baseline == 'GBDT':
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        if good_rest!=[]:
            good_rest = scaler.fit_transform(good_rest)
    else:
        scaler = MinMaxScaler()
        x = scaler.fit_transform(x.T).T    
        x = x.reshape((x.shape[0],x.shape[1],1))
        if good_rest!=[]:
            good_rest = scaler.fit_transform(good_rest.T).T
            good_rest = good_rest.reshape((good_rest.shape[0],good_rest.shape[1],1))
        
    indices = list(range(len(x)))  
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]  
    if good_rest!=[]:
        indices = list(range(len(good_rest)))  
        np.random.shuffle(indices)
        good_rest = good_rest[indices]                  
    return x,y,good_rest

def readdata(speeds,health,dataset,baseline):
    good=[]
    bad=[]
    for i in range(len(speeds)):
        if baseline == 'ResNet':
            if health == 'Good':
                varname_good = 'a_' + dataset[0] + '_Good_' + speeds[i]
                good_speed = np.load('./Data/All_speeds/' + varname_good + '.npy')
            else:
                varname_bad = 'a_' + dataset[0] + '_Bad_' + speeds[i]                            
                bad_speed = np.load('./Data/All_speeds/' + varname_bad + '.npy')   
        elif baseline == 'CWT+ResNet':
            if health == 'Good':
                varname_good = 'a_' + dataset[0] + '_Good_' + speeds[i] + '_sacwt'                      
                good_speed = np.load('./Data/All_featurespace/' + varname_good + '.npy')
            else:
                varname_bad = 'a_' + dataset[0] + '_Bad_' + speeds[i]  + '_sacwt'   
                bad_speed = np.load('./Data/All_featurespace/' + varname_bad + '.npy')               
        elif baseline == 'HT+ResNet':
            if health == 'Good':
                varname_good = 'a_' + dataset[0] + '_Good_' + speeds[i] + '_env'
                good_speed = np.load('./Data/All_featurespace/' + varname_good + '.npy')
            else:
                varname_bad = 'a_' + dataset[0] + '_Bad_' + speeds[i]  + '_env'                
                bad_speed = np.load('./Data/All_featurespace/' + varname_bad + '.npy')   
        elif baseline == 'GBDT':
            if health == 'Good':
                varname_good = 'a_' + dataset[0] + '_Good_' + speeds[i] + '_sf'            
                good_speed = np.load('./Data/All_featurespace/' + varname_good + '.npy')
            else:
                varname_bad = 'a_' + dataset[0] + '_Bad_' + speeds[i]  + '_sf'
                bad_speed = np.load('./Data/All_featurespace/' + varname_bad + '.npy')      
        n_aug = 1000        
        if health == 'Good':            
            if len(good_speed)>n_aug:
                good.append(good_speed[:n_aug])
            else:
                good.append(good_speed)
        else:
            if len(bad_speed)>n_aug:
                bad.append(bad_speed[:n_aug])
            else:
                bad.append(bad_speed)
                
    if health == 'Good':    
        x=np.vstack(good)
        n = len(x)
        y=np.hstack((np.reshape(np.zeros(n),(n,1)),np.reshape(np.ones(n),(n,1)))) # 0 1 for good
    else:
        x=np.vstack(bad)
        n = len(x)
        y=np.hstack((np.reshape(np.ones(n),(n,1)),np.reshape(np.zeros(n),(n,1))))  # 1 0 for bad 
    
    if baseline == 'GBDT':
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
    else:
        x = x.reshape((x.shape[0],x.shape[1]))
        if x.shape[1]==1024:
            x = resample(x,10240,axis=1)
        scaler = MinMaxScaler()
        x = scaler.fit_transform(x.T).T    
        x = x.reshape((x.shape[0],x.shape[1],1))
        
    indices = list(range(len(x)))  
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]            
    return x,y   

# input parameters
dataset = ['ESZUEG']
speeds_train = ['v55v65','v65v75','v75v85','v85v95','v95v105'] 
speeds_test_0 = ['v15v25'] 
speeds_test_1 = ['v25v35']
speeds_test_2 = ['v35v45']
speeds_test_3 = ['v45v55']
baselines = ['ResNet','CWT+ResNet','HT+ResNet','GBDT']
sample_size = 10240
repeats = 10  
rs_init = 2
tf.random.set_random_seed(rs_init)

# MBS-FWFSA
dataset_aug = ['syn']
speeds_train_aug = ['v15','v20_new2k','v30_new2k']
# SIM-GAN
# dataset_aug = ['syn']
# speeds_train_aug = ['v20v50_simgan']
# # cDCGAN
# dataset_aug = ['syn']
# speeds_train_aug = ['v20v50_cdcgan']
# # SIM
# dataset_aug = ['sim']
# speeds_train_aug = ['v15','v20','v30']


acc_GBDT = np.zeros((10,5))
acc_ResNet = np.zeros((10,5))
acc_HT = np.zeros((10,5))
acc_CWT = np.zeros((10,5))
indics_false = []
for baseline in baselines:
    # baeline model
    if baseline == 'ResNet':
        model = ResNet((10240,1))
    elif baseline == 'CWT+ResNet':
        model = ResNet((10240,1))
    elif baseline == 'HT+ResNet':
        model = ResNet_env((256,1))
    elif baseline == 'GBDT':
        model = GradientBoostingClassifier(learning_rate=0.05, n_estimators=200)

    for n in range(repeats):
        # get the balanced data and label the data
        x_train, y_train, good_rest_train = data4balance(speeds_train,dataset,baseline)
        x_train, x_test, y_train, y_test  = train_test_split(x_train, y_train, test_size=0.2, random_state=rs_init)
        # add synthetic WF data samples and some real healthy data for balance
        x_train_aug, y_train_aug = readdata(speeds_train_aug,'Bad',dataset_aug,baseline)
        # x_train_aug, y_train_aug,_ = data4balance(speeds_train_aug,dataset_aug,baseline)
        # n_aug = len(x_train_aug)
        # x_train_good = good_rest_train
        # y_train_good = np.hstack((np.reshape(np.zeros(n_aug),(n_aug,1)),np.reshape(np.ones(n_aug),(n_aug,1)))) # 0 1 for good
        # mix the train data        
        # x_train = np.vstack((x_train,x_train_aug,x_train_good[:n_aug]))
        # y_train = np.vstack((y_train,y_train_aug,y_train_good[:n_aug]))
        
        x_train = np.vstack((x_train,x_train_aug))
        y_train = np.vstack((y_train,y_train_aug))        
        indices = list(range(len(x_train)))  
        np.random.shuffle(indices)
        x_train = x_train[indices]
        y_train = y_train[indices] 
        
        x_test_0, y_test_0, _ = data4balance(speeds_test_0,dataset,baseline)
        x_test_1, y_test_1, _ = data4balance(speeds_test_1,dataset,baseline)
        x_test_2, y_test_2, _ = data4balance(speeds_test_2,dataset,baseline)
        x_test_3, y_test_3, _ = data4balance(speeds_test_3,dataset,baseline)             
        
        if baseline == 'GBDT':
            model.fit(x_train, y_train[:,0])
            acc_GBDT[n,0],_=test_GBDT(model,x_test,y_test)
            acc_GBDT[n,1],_=test_GBDT(model,x_test_0,y_test_0)
            acc_GBDT[n,2],_=test_GBDT(model,x_test_1,y_test_1)
            acc_GBDT[n,3],_=test_GBDT(model,x_test_2,y_test_2)
            acc_GBDT[n,4],_=test_GBDT(model,x_test_3,y_test_3)
        else:
            history = model.fit(x_train, y_train, batch_size=32,epochs=30, validation_split=0.2)
            if baseline == 'ResNet':
                acc_ResNet[n,0], _ = test_CNN(model,x_test,y_test)
                acc_ResNet[n,1], temp = test_CNN(model,x_test_0,y_test_0)
                acc_ResNet[n,2], _ = test_CNN(model,x_test_1,y_test_1)
                acc_ResNet[n,3], _ = test_CNN(model,x_test_2,y_test_2)
                acc_ResNet[n,4], _ = test_CNN(model,x_test_3,y_test_3)
                indics_false.append(temp)
            elif baseline == 'CWT+ResNet':
                acc_CWT[n,0],_=test_CNN(model,x_test,y_test)
                acc_CWT[n,1],_=test_CNN(model,x_test_0,y_test_0)
                acc_CWT[n,2],_=test_CNN(model,x_test_1,y_test_1)
                acc_CWT[n,3],_=test_CNN(model,x_test_2,y_test_2)
                acc_CWT[n,4],_=test_CNN(model,x_test_3,y_test_3)  
            elif baseline == 'HT+ResNet':
                acc_HT[n,0],_=test_CNN(model,x_test,y_test)
                acc_HT[n,1],_=test_CNN(model,x_test_0,y_test_0)
                acc_HT[n,2],_=test_CNN(model,x_test_1,y_test_1)
                acc_HT[n,3],_=test_CNN(model,x_test_2,y_test_2)
                acc_HT[n,4],_=test_CNN(model,x_test_3,y_test_3) 
                model.save('./results/HTResNet_aug_SpeedVariation')                
        rs_init += np.random.randint(1,50)
    