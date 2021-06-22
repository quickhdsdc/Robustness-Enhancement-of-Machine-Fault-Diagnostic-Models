import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from models.ResNet import test_CNN, test_GBDT, ResNet, ResNet_env, ResNet_CWT
from scipy.signal import resample
#
def data4balance(speeds,dataset,baseline):
    good=[]
    bad=[]
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
        
        n_aug = 200 # for FWFSA and SIM
        # n_aug = 600 # for SIM-GAN and cDCGAN
        if (dataset[0] == 'syn') or (dataset[0] == 'sim'):
            good.append(good_speed[:n_aug])
            bad.append(bad_speed[:n_aug])
        else:
            good.append(good_speed)
            bad.append(bad_speed)                  
        
    good=np.vstack(good)
    n = len(good)
    label_good=np.hstack((np.reshape(np.zeros(n),(n,1)),np.reshape(np.ones(n),(n,1)))) # 0 1 for good
    bad=np.vstack(bad)
    n = len(bad)
    label_bad=np.hstack((np.reshape(np.ones(n),(n,1)),np.reshape(np.zeros(n),(n,1))))  # 1 0 for bad 
    
    x = np.vstack((good,bad))
    y = np.vstack((label_good,label_bad))
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
#
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
        
        n_aug = 300        
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
# baselines = ['ResNet','CWT+ResNet','HT+ResNet','GBDT']
baselines = ['HT+ResNet']
dataset = ['ESZUEG']
speeds_train = ['v35v75_clean']
speeds_test_0 = ['v35v75_WFbad']
speeds_test_1 = ['v35v75_osc']
speeds_test_2 = ['v35v75_peak']
speeds_test_3 = ['v35v75_per']
sample_size = 10240
repeats = 1
rs_init = 1


#all
dataset_aug = ['syn']
speeds_train_aug = ['v40_interf','v50_interf','v70_interf','v35v75_simgan','v35v75_cdcgan']
# MBS-FWFSA
dataset_aug = ['syn']
speeds_train_aug = ['v40_interf','v50_interf','v70_interf']
# SIM-GAN
# dataset_aug = ['syn']
# speeds_train_aug = ['v35v75_simgan']
# # cDCGAN
# dataset_aug = ['syn']
# speeds_train_aug = ['v35v75_cdcgan']
# # SIM
# dataset_aug = ['sim']
# speeds_train_aug = ['v40','v50','v70']


acc_GBDT = np.zeros((10,5))
acc_ResNet = np.zeros((10,5))
acc_HT = np.zeros((10,5))
acc_CWT = np.zeros((10,5))
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
        x_train, y_train = data4balance(speeds_train,dataset,baseline)
        x_train, x_test, y_train, y_test  = train_test_split(x_train, y_train, test_size=0.2, random_state=rs_init)
        x_train_aug, y_train_aug = data4balance(speeds_train_aug,dataset_aug,baseline)
        x_train = np.vstack((x_train,x_train_aug))
        y_train = np.vstack((y_train,y_train_aug))   
        
        dataset_aug = ['sim']
        speeds_train_aug = ['v40','v50','v70']        
        x_train_aug, y_train_aug = data4balance(speeds_train_aug,dataset_aug,baseline)      
        x_train = np.vstack((x_train,x_train_aug))
        y_train = np.vstack((y_train,y_train_aug))          
        
        indices = list(range(len(x_train)))  
        np.random.shuffle(indices)
        x_train = x_train[indices]
        y_train = y_train[indices]  
        
        x_test_0, y_test_0 = readdata(speeds_test_0,'Bad',dataset,baseline)
        x_test_1, y_test_1 = readdata(speeds_test_1,'Good',dataset,baseline)
        x_test_2, y_test_2 = readdata(speeds_test_2,'Good',dataset,baseline)     
        x_test_3, y_test_3 = readdata(speeds_test_3,'Good',dataset,baseline)
        
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
                acc_ResNet[n,0],_=test_CNN(model,x_test,y_test)
                acc_ResNet[n,1],_=test_CNN(model,x_test_0,y_test_0)
                acc_ResNet[n,2],_=test_CNN(model,x_test_1,y_test_1)
                acc_ResNet[n,3],_=test_CNN(model,x_test_2,y_test_2)
                acc_ResNet[n,4],_=test_CNN(model,x_test_3,y_test_3)
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
                model.save('./results/HTResNet_aug_InterferenceVariation')
        rs_init += np.random.randint(1,50)
    