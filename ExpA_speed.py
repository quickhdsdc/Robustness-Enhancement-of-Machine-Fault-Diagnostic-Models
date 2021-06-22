import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from Models.ResNet import test_CNN, test_GBDT, ResNet, ResNet_env
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
        n_good = len(good_speed)        
        n_bad = len(bad_speed)        
        
        if n_good>=n_bad:    
            indices = list(range(n_good))   
            np.random.shuffle(indices)
            idx_good = indices[:n_bad]
            good_speed = good_speed[idx_good]
        else:
            indices = list(range(n_bad))   
            np.random.shuffle(indices)
            idx_bad = indices[:n_good]
            bad_speed = bad_speed[idx_bad]
        
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
rs_init = 1

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
        x_test_0, y_test_0 = data4balance(speeds_test_0,dataset,baseline)
        x_test_1, y_test_1 = data4balance(speeds_test_1,dataset,baseline)
        x_test_2, y_test_2 = data4balance(speeds_test_2,dataset,baseline)
        x_test_3, y_test_3 = data4balance(speeds_test_3,dataset,baseline)             
        x_train, x_test, y_train, y_test  = train_test_split(x_train, y_train, test_size=0.2, random_state=rs_init)
        if baseline == 'GBDT':
            model.fit(x_train, y_train[:,0])
            acc_GBDT[n,0],_,_,_,_=test_GBDT(model,x_test,y_test)
            acc_GBDT[n,1],_,_,_,_=test_GBDT(model,x_test_0,y_test_0)
            acc_GBDT[n,2],_,_,_,_=test_GBDT(model,x_test_1,y_test_1)
            acc_GBDT[n,3],_,_,_,_=test_GBDT(model,x_test_2,y_test_2)
            acc_GBDT[n,4],_,_,_,_=test_GBDT(model,x_test_3,y_test_3)
        else:
            history = model.fit(x_train, y_train, batch_size=64,epochs=30, validation_split=0.2)
            if baseline == 'ResNet':
                acc_ResNet[n,0],_,_,_,_=test_CNN(model,x_test,y_test)
                acc_ResNet[n,1],_,_,_,_=test_CNN(model,x_test_0,y_test_0)
                acc_ResNet[n,2],_,_,_,_=test_CNN(model,x_test_1,y_test_1)
                acc_ResNet[n,3],_,_,_,_=test_CNN(model,x_test_2,y_test_2)
                acc_ResNet[n,4],_,_,_,_=test_CNN(model,x_test_3,y_test_3)
            elif baseline == 'CWT+ResNet':
                acc_CWT[n,0],_,_,_,_=test_CNN(model,x_test,y_test)
                acc_CWT[n,1],_,_,_,_=test_CNN(model,x_test_0,y_test_0)
                acc_CWT[n,2],_,_,_,_=test_CNN(model,x_test_1,y_test_1)
                acc_CWT[n,3],_,_,_,_=test_CNN(model,x_test_2,y_test_2)
                acc_CWT[n,4],_,_,_,_=test_CNN(model,x_test_3,y_test_3)  
            elif baseline == 'HT+ResNet':
                acc_HT[n,0],_,_,_,_=test_CNN(model,x_test,y_test)
                acc_HT[n,1],_,_,_,_=test_CNN(model,x_test_0,y_test_0)
                acc_HT[n,2],_,_,_,_=test_CNN(model,x_test_1,y_test_1)
                acc_HT[n,3],_,_,_,_=test_CNN(model,x_test_2,y_test_2)
                acc_HT[n,4],_,_,_,_=test_CNN(model,x_test_3,y_test_3)                 
        rs_init += np.random.randint(1,50)
