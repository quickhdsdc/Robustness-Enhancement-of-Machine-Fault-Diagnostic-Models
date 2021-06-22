import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from models.ResNet import test_CNN, test_GBDT, ResNet, ResNet_env, ResNet_CWT
#
def data4balance(speeds,dataset,baseline):
    good=[]
    bad=[]
    for i in range(len(speeds)):
        if baseline == 'ResNet':
            varname_good = 'a_ESZUEG_Good_' + speeds[i] + '_' + dataset[0]
            varname_bad = 'a_ESZUEG_Bad_' + speeds[i] + '_' + dataset[0]            
            good_speed = np.load('./Data/Axlebox_trackirr/' + varname_good + '.npy')
            bad_speed = np.load('./Data/Axlebox_trackirr/' + varname_bad + '.npy')   
        elif baseline == 'CWT+ResNet':
            varname_good = 'a_ESZUEG_Good_' + speeds[i] + '_' + dataset[0] + '_sacwt'
            varname_bad = 'a_ESZUEG_Bad_' + speeds[i] + '_' + dataset[0]  + '_sacwt'         
            good_speed = np.load('./Data/All_featurespace/' + varname_good + '.npy')
            bad_speed = np.load('./Data/All_featurespace/' + varname_bad + '.npy')               
        elif baseline == 'HT+ResNet':
            varname_good = 'a_ESZUEG_Good_' + speeds[i] + '_' + dataset[0] + '_env'
            varname_bad = 'a_ESZUEG_Bad_' + speeds[i] + '_' + dataset[0]  + '_env'         
            good_speed = np.load('./Data/All_featurespace/' + varname_good + '.npy')
            bad_speed = np.load('./Data/All_featurespace/' + varname_bad + '.npy')   
        elif baseline == 'GBDT':
            varname_good = 'a_ESZUEG_Good_' + speeds[i] + '_' + dataset[0] + '_sf'
            varname_bad = 'a_ESZUEG_Bad_' + speeds[i] + '_' + dataset[0]  + '_sf'         
            good_speed = np.load('./Data/All_featurespace/' + varname_good + '.npy')
            bad_speed = np.load('./Data/All_featurespace/' + varname_bad + '.npy')                   
        
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


# input parameters
baselines = ['ResNet','CWT+ResNet','HT+ResNet','GBDT']
# baselines = ['GBDT']
dataset_train = ['clean']
dataset_test_0 = ['WFbad']
dataset_test_1 = ['osc']
dataset_test_2 = ['peak']
dataset_test_3 = ['per']
speeds = ['v35v75'] 
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
        x_train, y_train = data4balance(speeds,dataset_train,baseline)
        x_test_0, y_test_0 = readdata(speeds,'Bad',dataset_test_0,baseline)
        x_test_1, y_test_1 = readdata(speeds,'Good',dataset_test_1,baseline)
        x_test_2, y_test_2 = readdata(speeds,'Good',dataset_test_2,baseline)     
        x_test_3, y_test_3 = readdata(speeds,'Good',dataset_test_3,baseline)
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
    