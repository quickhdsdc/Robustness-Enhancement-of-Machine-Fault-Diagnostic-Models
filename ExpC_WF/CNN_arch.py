#=============================================================================#
############################# import functions ################################
#=============================================================================#
from keras.layers import GlobalAveragePooling1D, Multiply, Dense
from keras.models import Model
from keras.layers import Conv1D, Dense, Flatten, Activation, Input, BatchNormalization, Dropout, Reshape
from keras.layers import Activation, BatchNormalization, MaxPooling1D, AveragePooling1D, SeparableConv1D, Add, GlobalAveragePooling1D
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
import tensorflow as tf
from keras.layers import Layer
import math
from keras import backend as K
from keras.optimizers import Adam
import os
import numpy as np
from Mish import Mish, Relu6, Hswish
from lookahead import Lookahead
from sklearn.metrics import roc_curve
from sklearn import metrics
import matplotlib.pyplot as plt
from FeatureExtraction import wpt,stft_powerspectrum,ExtraFeatures,LSSVM
import pywt
import pandas as pd
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier 
#=============================================================================#
################################ Blocks of CNN ################################
#=============================================================================#
def conv_block(x, filters, kernel_size, strides, se, ratio, act, name):
    y = Conv1D(filters, kernel_size, padding='same', strides=strides,  kernel_initializer='VarianceScaling', name='{}_conv'.format(name))(x)
    if se:
        y = squeezeExcite(y, ratio,name=name)
    y = BatchNormalization(name='{}_bn'.format(name))(y)        
    y = Activation(act, name='{}_act'.format(name))(y)    
    return y

def squeezeExcite(x, ratio, name):
    nb_chan = K.int_shape(x)[-1]
    y = GlobalAveragePooling1D(name='{}_se_avg'.format(name))(x)
    y = Dense(nb_chan // ratio, activation='relu', name='{}_se_dense1'.format(name))(y)
    y = Dense(nb_chan, activation='hard_sigmoid', name='{}_se_dense2'.format(name))(y)
    y = Reshape((1, nb_chan))(y)
    y = Multiply(name='{}_se_mul'.format(name))([x, y])
    return y

def separableconv_block(x, filters, kernel_size, strides, se, ratio, act, name):
    y = SeparableConv1D(filters=filters, kernel_size=kernel_size, padding='same', strides=strides, kernel_initializer='VarianceScaling', name='{}_separableconv'.format(name))(x)
    if se:
        y = squeezeExcite(y, ratio, name='{}_se'.format(name))
    y = BatchNormalization(name='{}_bn'.format(name))(y)        
    y = Activation(act, name='{}_act'.format(name))(y)    
    return y

def bottleneck(x, filters, kernel_size, expansion, strides, se, ratio, act, name):
    channel_axis = -1
    in_channels = K.int_shape(x)[channel_axis]   
    y=conv_block(x, filters=filters//expansion, kernel_size=1, strides=1, se=False, ratio=8, act=act, name='{}_conv'.format(name))
    y = SeparableConv1D(filters=filters, kernel_size=kernel_size, padding='same', strides=strides, name='{}_separableconv'.format(name))(y)
    if se:
        y = squeezeExcite(y,ratio,name=name)    
    y = BatchNormalization(name='{}_bn'.format(name))(y)            
    if filters==in_channels and strides==1:
        y = Add(name='{}_Projectadd'.format(name))([x,y])
    return y
    
def ResBlockv2(x, filters, kernel_size, strides, ratio, act, name):
    channel_axis = -1
    in_channels = K.int_shape(x)[channel_axis]  
    y = BatchNormalization()(x)        
    y = Activation(act)(y)
    y = Conv1D(filters,kernel_size,padding='same',strides=strides,name='{}conv1'.format(name))(y)
    y = BatchNormalization()(y)        
    y = Activation(act)(y)    
    y = Conv1D(filters,kernel_size,padding='same',strides=strides,name='{}conv2'.format(name))(y)
    if filters==in_channels and strides==1:
        y = Add(name='{}_Projectadd'.format(name))([x,y])   
    return y

def ResBlockv2_2D(x, filters, kernel_size, strides, ratio, act, name):
    channel_axis = -1
    in_channels = K.int_shape(x)[channel_axis]  
    y = BatchNormalization()(x)        
    y = Activation(act)(y)
    y = Conv2D(filters,kernel_size,padding='same',strides=strides,name='{}conv1'.format(name))(y)
    y = BatchNormalization()(y)        
    y = Activation(act)(y)    
    y = Conv2D(filters,kernel_size,padding='same',strides=strides,name='{}conv2'.format(name))(y)
    if filters==in_channels and strides==1:
        y = Add(name='{}_Projectadd'.format(name))([x,y])   
    return y

#=============================================================================#
#################### Build a CNN model: ResNet TSclassification ###############
#=============================================================================#
#def CNN_arch(input_shape):
#    n_feature_maps = 64
#    input_layer = Input(input_shape)   
#    # BLOCK 1 
#    conv_x = Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
#    conv_x = BatchNormalization()(conv_x)
#    conv_x = Activation('relu')(conv_x)
#    conv_y = Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
#    conv_y = BatchNormalization()(conv_y)
#    conv_y = Activation('relu')(conv_y)
#    conv_z = Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
#    conv_z = BatchNormalization()(conv_z)
#    # expand channels for the sum 
#    shortcut_y = Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
#    shortcut_y = BatchNormalization()(shortcut_y)
#    output_block_1 = Add()([shortcut_y, conv_z])
#    output_block_1 = Activation('relu')(output_block_1)
#    # BLOCK 2 
#    conv_x = Conv1D(filters=n_feature_maps*2, kernel_size=8, padding='same')(output_block_1)
#    conv_x = BatchNormalization()(conv_x)
#    conv_x = Activation('relu')(conv_x)
#    conv_y = Conv1D(filters=n_feature_maps*2, kernel_size=5, padding='same')(conv_x)
#    conv_y = BatchNormalization()(conv_y)
#    conv_y = Activation('relu')(conv_y)
#    conv_z = Conv1D(filters=n_feature_maps*2, kernel_size=3, padding='same')(conv_y)
#    conv_z = BatchNormalization()(conv_z)
#    # expand channels for the sum 
#    shortcut_y = Conv1D(filters=n_feature_maps*2, kernel_size=1, padding='same')(output_block_1)
#    shortcut_y = BatchNormalization()(shortcut_y)
#    output_block_2 = Add()([shortcut_y, conv_z])
#    output_block_2 = Activation('relu')(output_block_2)
#    # BLOCK 3 
#    conv_x = Conv1D(filters=n_feature_maps*2, kernel_size=8, padding='same')(output_block_2)
#    conv_x = BatchNormalization()(conv_x)
#    conv_x = Activation('relu')(conv_x)
#    conv_y = Conv1D(filters=n_feature_maps*2, kernel_size=5, padding='same')(conv_x)
#    conv_y = BatchNormalization()(conv_y)
#    conv_y = Activation('relu')(conv_y)
#    conv_z = Conv1D(filters=n_feature_maps*2, kernel_size=3, padding='same')(conv_y)
#    conv_z = BatchNormalization()(conv_z)
#    # no need to expand channels because they are equal 
#    shortcut_y = BatchNormalization()(output_block_2)
#    output_block_3 = Add()([shortcut_y, conv_z])
#    output_block_3 = Activation('relu')(output_block_3)
#    # FINAL     
#    gap_layer = GlobalAveragePooling1D()(output_block_3)
#    output_layer = Dense(2, activation='softmax')(gap_layer)
#    model = Model(inputs=input_layer, outputs=output_layer)
#    model.compile(optimizer=Adam(lr=0.001), loss = 'binary_crossentropy', metrics = ['accuracy'])
#    return model
#=============================================================================#
###################### Build a CNN model: 1DResNet ###########################
#=============================================================================#
# input(10000,1)
#def CNN_arch(input_shape):
#    data_input = Input(shape=input_shape)
#    x = conv_block(data_input, filters=8, kernel_size=3, strides=2, se=False, ratio=8, act='relu', name='block1')  
#    x = conv_block(data_input, filters=8, kernel_size=3, strides=2, se=False, ratio=8, act='relu', name='block2')     
#    x = MaxPooling1D(pool_size=10, strides=10, padding='same')(x)
#
#    x=ResBlockv2(x, filters=16, kernel_size=3, strides=2, act='relu', ratio=8, name='block31')    
#    x=ResBlockv2(x, filters=16, kernel_size=3, strides=1, act='relu', ratio=8, name='block32')
#    x=ResBlockv2(x, filters=16, kernel_size=3, strides=1, act='relu', ratio=8, name='block33')   
##    x = bottleneck(x, filters=16, kernel_size=3, expansion=4, strides=2,  se=True, ratio=8, act='Hswish', name='block3')    
##    x = bottleneck(x, filters=16, kernel_size=3, expansion=4, strides=1,  se=True, ratio=8, act='Hswish', name='block4')    
#     
#    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)
#    
#    x = GlobalAveragePooling1D()(x) 
#    x = Dense(2, kernel_initializer='VarianceScaling', activation='softmax')(x)   
#    model = Model(inputs=data_input, outputs=x)
#    model.compile(optimizer=Adam(lr=0.001), loss = 'binary_crossentropy', metrics = ['accuracy'])
#    return model
#=============================================================================#
###################### Build a CNN model: 1DResNet ###########################
#=============================================================================#
#input(1000,1)
#def CNN_arch(input_shape):
#    data_input = Input(shape=input_shape)
#    x = conv_block(data_input, filters=8, kernel_size=3, strides=2, se=False, ratio=8, act='relu', name='block1')  
#    x = conv_block(data_input, filters=8, kernel_size=3, strides=2, se=False, ratio=8, act='relu', name='block2')     
#    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)
#
#    x=ResBlockv2(x, filters=16, kernel_size=3, strides=2, act='relu', ratio=8, name='block31')    
#    x=ResBlockv2(x, filters=16, kernel_size=3, strides=1, act='relu', ratio=8, name='block32')
##    x=ResBlockv2(x, filters=16, kernel_size=3, strides=1, act='relu', ratio=8, name='block33')   
##    x = bottleneck(x, filters=16, kernel_size=3, expansion=4, strides=2,  se=True, ratio=8, act='Hswish', name='block3')    
##    x = bottleneck(x, filters=16, kernel_size=3, expansion=4, strides=1,  se=True, ratio=8, act='Hswish', name='block4')    
#     
#    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)
#    
#    x = GlobalAveragePooling1D()(x) 
#    x = Dense(2, kernel_initializer='VarianceScaling', activation='softmax')(x)   
#    model = Model(inputs=data_input, outputs=x)
#    model.compile(optimizer=Adam(lr=0.001), loss = 'binary_crossentropy', metrics = ['accuracy'])
#    return model
#=============================================================================#
################### Build a CNN model: WPT + 1DResNet #########################
#=============================================================================#
#input(1000,1)
#def CNN_arch(input_shape):
#    data_input = Input(shape=input_shape)
#    x = conv_block(data_input, filters=8, kernel_size=3, strides=1, se=False, ratio=8, act='relu', name='block1')  
#    x = conv_block(data_input, filters=8, kernel_size=3, strides=1, se=False, ratio=8, act='relu', name='block2')     
#    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)
#
#    x=ResBlockv2(x, filters=16, kernel_size=3, strides=2, act='relu', ratio=8, name='block31')    
#    x=ResBlockv2(x, filters=16, kernel_size=3, strides=1, act='relu', ratio=8, name='block32')
#     
#    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)
#    
#    x = GlobalAveragePooling1D()(x) 
#    x = Dense(2, kernel_initializer='VarianceScaling', activation='softmax')(x)   
#    model = Model(inputs=data_input, outputs=x)
#    model.compile(optimizer=Adam(lr=0.001), loss = 'binary_crossentropy', metrics = ['accuracy'])
#    return model
#=============================================================================#
################ Build a CNN model: STFT + 2DResNet ###########################
#=============================================================================#
#input(1000,1)
#def CNN_arch(input_shape):
#    data_input = Input(shape=input_shape)
#    x = Conv2D(filters=8, kernel_size=3, strides=1,name='block11')(data_input)
#    x = Conv2D(filters=8, kernel_size=3, strides=1,name='block12')(data_input)    
#    x = MaxPooling2D(pool_size=2, strides=2, padding='same')(x)
#
#    x=ResBlockv2_2D(x, filters=16, kernel_size=3, strides=2, act='relu', ratio=8, name='block21')    
#    x=ResBlockv2_2D(x, filters=16, kernel_size=3, strides=1, act='relu', ratio=8, name='block22')
##    x=ResBlockv2_2D(x, filters=16, kernel_size=3, strides=1, act='relu', ratio=8, name='block23')
#    
#    x = MaxPooling2D(pool_size=2, strides=2, padding='same')(x)
#    
#    x = GlobalAveragePooling2D()(x) 
#    x = Dense(2, kernel_initializer='VarianceScaling', activation='softmax')(x)   
#    model = Model(inputs=data_input, outputs=x)
#    model.compile(optimizer=Adam(lr=0.001), loss = 'binary_crossentropy', metrics = ['accuracy'])
#    return model
#=============================================================================#
################### Build a CNN model: WPT + 1DResNet #########################
#=============================================================================#
# input(10000,1)
#def CNN_arch(input_shape):
#    data_input = Input(shape=input_shape)
#    x = conv_block(data_input, filters=8, kernel_size=3, strides=2, se=False, ratio=8, act='relu', name='block1')  
#    x = conv_block(data_input, filters=8, kernel_size=3, strides=2, se=False, ratio=8, act='relu', name='block2')     
#    x = MaxPooling1D(pool_size=8, strides=8, padding='same')(x)
#
#    x=ResBlockv2(x, filters=16, kernel_size=3, strides=2, act='relu', ratio=8, name='block31')    
#    x=ResBlockv2(x, filters=16, kernel_size=3, strides=1, act='relu', ratio=8, name='block32')
#    x=ResBlockv2(x, filters=16, kernel_size=3, strides=1, act='relu', ratio=8, name='block33')   
##    x = bottleneck(x, filters=16, kernel_size=3, expansion=4, strides=2,  se=True, ratio=8, act='Hswish', name='block3')    
##    x = bottleneck(x, filters=16, kernel_size=3, expansion=4, strides=1,  se=True, ratio=8, act='Hswish', name='block4')    
#     
#    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)
#    
#    x = GlobalAveragePooling1D()(x) 
#    x = Dense(2, kernel_initializer='VarianceScaling', activation='softmax')(x)   
#    model = Model(inputs=data_input, outputs=x)
#    model.compile(optimizer=Adam(lr=0.001), loss = 'binary_crossentropy', metrics = ['accuracy'])
#    return model
#=============================================================================#
################ Build a CNN model: STFT + 2DResNet ###########################
#=============================================================================#
# input(10000,1)
#def CNN_arch(input_shape):
#    data_input = Input(shape=input_shape)
#    x = Conv2D(filters=8, kernel_size=3, strides=2,name='block11')(data_input)
#    x = Conv2D(filters=8, kernel_size=3, strides=2,name='block12')(data_input)    
#    x = MaxPooling2D(pool_size=2, strides=2, padding='same')(x)
#
#    x=ResBlockv2_2D(x, filters=16, kernel_size=3, strides=2, act='relu', ratio=8, name='block21')    
#    x=ResBlockv2_2D(x, filters=16, kernel_size=3, strides=1, act='relu', ratio=8, name='block22')
#    x=ResBlockv2_2D(x, filters=16, kernel_size=3, strides=1, act='relu', ratio=8, name='block23')
#    
#    x = MaxPooling2D(pool_size=2, strides=2, padding='same')(x)
#    
#    x = GlobalAveragePooling2D()(x) 
#    x = Dense(2, kernel_initializer='VarianceScaling', activation='softmax')(x)   
#    model = Model(inputs=data_input, outputs=x)
#    model.compile(optimizer=Adam(lr=0.001), loss = 'binary_crossentropy', metrics = ['accuracy'])
#    return model

#=============================================================================#
############################ test function ####################################
#=============================================================================#  
#def testing(model,x_test,y_test):  
##    test_results = model.evaluate(x_test, y_test)
##    for name, value in zip(model.metrics_names, test_results):
##        print(name, value)  
#    
#    y_pred = model.predict(x_test)  
##    y_pred = model.predict_proba(x_test)    
#    cls_pre = np.argmin(y_pred,axis=1)
#    cls_true = np.argmin(y_test,axis=1)
#    
#    indics_false = np.where(cls_pre!=cls_true)[0]
#    indics_true = np.where(cls_pre==cls_true)[0]
#    
#    population = len(cls_true)
#    # positive = bad = wheel flat # negative = good = intact wheel
#    # TP = coorectly diagnosis # FP = false alarm # FN = non-success diagnosis
#    # recall reveals the cablability of diagnosis
#    # precision indicates the avoidancy of false alarm
#    T = len(indics_true)
#    F = len(indics_false)
#    TP = np.count_nonzero(cls_pre[indics_true])
#    TN = T-TP
#    FP = np.count_nonzero(cls_pre[indics_false]) #
#    FN = F-FP   
#    TPR = TP/(TP+FN)
#    TNR = TN/(TN+FP)
#    bACC = (TPR+TNR)/2
#    acc = T/population
#    recall = TP/(TP+FN)
#    precision = TP/(TP+FP)
##    f1 = 2*recall*precision/(recall+precision)
#    fpr, tpr, _ = roc_curve(cls_true, cls_pre)
#    auc= metrics.auc(fpr, tpr) 
#    print('bACC=%.2f' %(bACC*100)+'%')
#    print('recall=%.2f' %(recall*100)+'%')
#    print('precision=%.2f' %(precision*100)+'%')
##    print('f1 score=%.2f' %(f1*100)+'%')
#    print('AUC=%.2f' %(auc*100)+'%')
#    return bACC,recall,precision,auc,cls_pre,cls_true,indics_false,indics_true

def testing(model,x_test,y_test):  
#    test_results = model.evaluate(x_test, y_test)
#    for name, value in zip(model.metrics_names, test_results):
#        print(name, value)  
    
#    y_pred = model.predict(x_test)  
    y_pred = model.predict_proba(x_test)    
    cls_pre = np.argmax(y_pred,axis=1)
    cls_true = np.argmax(y_test,axis=1)
    
    indics_false = np.where(cls_pre!=cls_true)[0]
    indics_true = np.where(cls_pre==cls_true)[0]
    
    population = len(cls_true)
    # positive = bad = wheel flat # negative = good = intact wheel
    # TP = coorectly diagnosis # FP = false alarm # FN = non-success diagnosis
    # recall reveals the cablability of diagnosis
    # precision indicates the avoidancy of false alarm
    T = len(indics_true)
    F = len(indics_false)
    TP = len(cls_pre[indics_true]) - np.count_nonzero(cls_pre[indics_true])
    FP = len(cls_pre[indics_false])- np.count_nonzero(cls_pre[indics_false]) #
    FN = F-FP   
    acc = T/population
#    recall = TP/(TP+FN)
#    precision = TP/(TP+FP)
#    f1 = 2*recall*precision/(recall+precision)
#    fpr, tpr, _ = roc_curve(cls_true, cls_pre)
#    auc= metrics.auc(fpr, tpr) 
    print('acc=%.2f' %(acc*100)+'%')
#    print('recall=%.2f' %(recall*100)+'%')
#    print('precision=%.2f' %(precision*100)+'%')
##    print('f1 score=%.2f' %(f1*100)+'%')
#    print('AUC=%.2f' %(auc*100)+'%')
    return acc,cls_pre,cls_true,indics_false,indics_true
#=============================================================================#
###################### test the CNN model: WF condition #######################
#=============================================================================#
FolderPath_train = os.path.abspath(os.path.join(os.getcwd(), "./data/train_ESZUEG"))
for file in os.listdir(FolderPath_train):
    if file.endswith('.npy'):
        VarName, ext = os.path.splitext(file)
        FilePath = os.path.join(FolderPath_train, file)
        locals()[VarName] = np.load(FilePath)

#=============================================================================#
###################### add synthetic data #####################################
#=============================================================================#
n_syn=500
x_train_syn=np.zeros((n_syn*6,10000,1))  
y_train_syn=np.zeros((n_syn*6,2))
count=0
FolderPath_syn = os.path.abspath(os.path.join(os.getcwd(), "../data"))
FolderPath1 = ['v50','v20','v70']
FolderPath2 = ['WF','normal']
for i in range(len(FolderPath1)):
    for j in range(len(FolderPath2)):
        FileName = 'syn_'+ FolderPath2[j] + '_' + FolderPath1[i] + '.npy'
        FilePath = os.path.join(FolderPath_syn, FileName)
        VarName = 'syn_'+ FolderPath2[j] + '_' + FolderPath1[i]
        globals()[VarName] = np.load(FilePath)                
        count=count+1
        x_train_syn[int((count-1)*n_syn):int((count)*n_syn)]=np.reshape(globals()[VarName][:n_syn],(n_syn,10000,1))
        if FolderPath2[j]=='normal':
            y_train_syn[int((count-1)*n_syn):int((count)*n_syn),1]=1
        else:
            y_train_syn[int((count-1)*n_syn):int((count)*n_syn),0]=1
                  
x_train=np.append(x_train,x_train_syn,axis=0)        
y_train=np.append(y_train,y_train_syn,axis=0)  

#=============================================================================#
###################### pre-processing #########################################
#=============================================================================#
FolderPath_train = os.path.abspath(os.path.join(os.getcwd(), "./data/train_ESZUEG"))
for file in os.listdir(FolderPath_train):
    if file.endswith('.csv'):
        VarName, ext = os.path.splitext(file)
        FilePath = os.path.join(FolderPath_train, file)  
        locals()[VarName] = pd.read_csv(FilePath)
 
#FolderPath = os.path.abspath(os.path.join(os.getcwd(), "./data/"))
#for folder in os.listdir(FolderPath):
#    DataPath = os.path.join(FolderPath, folder)
#    for file in os.listdir(DataPath):
#        if file.endswith('.npy'):
#            VarName, ext = os.path.splitext(file)
#            FilePath = os.path.join(DataPath, file)  
#            globals()[VarName] = np.load(FilePath)
#    fs=5000
#    x_test=ExtraFeatures(x_test,fs)
#    csvpath_test= os.path.join(DataPath, 'x_test.csv')
#    x_test.to_csv(csvpath_test,index=False) 
#    try:
#        x_train=ExtraFeatures(x_train,fs)
#        csvpath_train= os.path.join(DataPath, 'x_train.csv')
#        x_train.to_csv(csvpath_train,index=False) 
#        
#        x_val=ExtraFeatures(x_val,fs)
#        csvpath_val= os.path.join(DataPath, 'x_val.csv')
#        x_val.to_csv(csvpath_val,index=False) 
#    except:
#        pass 

#fs=5000
#x_train=ExtraFeatures(x_train,fs)
#csvpath_train= os.path.join(FolderPath_train, 'x_train.csv')
#x_train.to_csv(csvpath_train,index=False) 

                
# wavelet paket transformation
#level=3
#x_train=wpt(x_train,level)
#x_val=wpt(x_val,level)
#x_test=wpt(x_test,level)
#width=len(x_train[0,:,0])


## STFT
#fs=5000    
#x_train=stft_powerspectrum(x_train,fs)
#x_val=stft_powerspectrum(x_val,fs)
#x_test=stft_powerspectrum(x_test,fs)  
        
# train and test
#FolderPath = ['train_ESZUEG_500','test_Entgleis','test_LEILA']
FolderPath = ['train_ESZUEG','test_Bosch']

metrics_train_ESZUEG=np.zeros((10,4))
metrics_test_Bosch=np.zeros((10,4))
#metrics_test_Entgleis=np.zeros((10,4))
#metrics_test_LEILA=np.zeros((10,4))


global best_criteria
best_criteria=0

for n in range(0,10):
    print('run: ' + str(n))    
    try:
        del model
        K.clear_session
    except:
        print('model is cleaned')
        pass   
#    model=CNN_arch((1256,8))
#    model=CNN_arch((80,80,1))
#    model=CNN_arch((131,8))
#    model=CNN_arch((33,33,1))        
#    model=CNN_arch((10000,1))                                                  # build model
#    history = model.fit(x_train,y_train,batch_size=32,epochs=20,validation_data=[x_val, y_val])  #fiting with hyperparameters
    model = GradientBoostingClassifier()
    model.fit(x_train.values, y_train[:,1])    
    for i in range(2):
        FolderPath_test = os.path.abspath(os.path.join(os.getcwd(), ("./data/"+FolderPath[i])))
        for file in os.listdir(FolderPath_test):
            if file.endswith('.npy'):
                VarName, ext = os.path.splitext(file)
                FilePath = os.path.join(FolderPath_test, file)
                locals()[VarName] = np.load(FilePath)

        FolderPath_test = os.path.abspath(os.path.join(os.getcwd(), ("./data/"+FolderPath[i])))
        for file in os.listdir(FolderPath_test):
            if file.endswith('.csv'):
                VarName, ext = os.path.splitext(file)
                FilePath = os.path.join(FolderPath_test, file)  
                locals()[VarName] = pd.read_csv(FilePath)   

#        if len(x_test[0,:,0])==10000:
#            x_test=wpt(x_test,level)                                               # wavelet paket transformation 
#        if len(x_train[0,:,0])==10000:
#            x_train=wpt(x_train,level)          
#        if len(x_val[0,:,0])==10000:
#            x_val=wpt(x_val,level)                  

#        if len(x_test[0,:,0])==10000:
#            x_test=stft_powerspectrum(x_test,fs)                                              # STFT 
#        if len(x_train[0,:,0])==10000:
#            x_train=stft_powerspectrum(x_train,fs)          
#        if len(x_val[0,:,0])==10000:
#            x_val=stft_powerspectrum(x_val,fs)  

        acc,_,_,_,_=testing(model,x_test,y_test)
        Varname_metrics = 'metrics_' + FolderPath[i]
        locals()[Varname_metrics][n,0]=acc
#        locals()[Varname_metrics][n,1]=recall
#        locals()[Varname_metrics][n,2]=precision
#        locals()[Varname_metrics][n,3]=auc        
#    criteria=np.mean(metrics_train_ESZUEG[n,0])
#    if criteria>best_criteria:
#        best_criteria=criteria
#        model.save('./CNN_AB_v15v105_run'+str(n))
#        np.save('train_acc_best',history.history['acc'])
#        np.save('val_acc_best',history.history['val_acc'])
#        np.save('train_loss_best',history.history['loss'])
#        np.save('val_loss_best',history.history['val_loss'])
        #summarize history for accuracy
#=============================================================================#
############# read train data and pre-processing again ########################
#=============================================================================#
    FolderPath_train = os.path.abspath(os.path.join(os.getcwd(), "./data/train_ESZUEG"))
    for file in os.listdir(FolderPath_train):
        if file.endswith('.npy'):
            VarName, ext = os.path.splitext(file)
            FilePath = os.path.join(FolderPath_train, file)
            locals()[VarName] = np.load(FilePath)    

    n_syn=500
    x_train_syn=np.zeros((n_syn*6,10000,1))  
    y_train_syn=np.zeros((n_syn*6,2))
    count=0
    FolderPath_syn = os.path.abspath(os.path.join(os.getcwd(), "../data"))
    FolderPath1 = ['v70','v50','v20']
    FolderPath2 = ['WF','normal']
    for i in range(len(FolderPath1)):
        for j in range(len(FolderPath2)):
            FileName = 'syn_'+ FolderPath2[j] + '_' + FolderPath1[i] + '.npy'
            FilePath = os.path.join(FolderPath_syn, FileName)
            VarName = 'syn_'+ FolderPath2[j] + '_' + FolderPath1[i]
            globals()[VarName] = np.load(FilePath)                
            count=count+1
            x_train_syn[int((count-1)*n_syn):int((count)*n_syn)]=np.reshape(globals()[VarName][:n_syn],(n_syn,10000,1))
            if FolderPath2[j]=='normal':
                y_train_syn[int((count-1)*n_syn):int((count)*n_syn),1]=1
            else:
                y_train_syn[int((count-1)*n_syn):int((count)*n_syn),0]=1
                      
    x_train=np.append(x_train,x_train_syn,axis=0)        
    y_train=np.append(y_train,y_train_syn,axis=0)  

#    x_train=wpt(x_train,level)
#    x_val=wpt(x_val,level)
#    x_test=wpt(x_test,level)
#    width=len(x_train[0,:,0])
            
#    x_test=stft_powerspectrum(x_test,fs)                                              # STFT 
#    x_train=stft_powerspectrum(x_train,fs)          
#    x_val=stft_powerspectrum(x_val,fs)  

    FolderPath_train = os.path.abspath(os.path.join(os.getcwd(), "./data/train_ESZUEG"))
    for file in os.listdir(FolderPath_train):
        if file.endswith('.csv'):
            VarName, ext = os.path.splitext(file)
            FilePath = os.path.join(FolderPath_train, file)  
            locals()[VarName] = pd.read_csv(FilePath)            