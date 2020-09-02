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
#def CNN_arch(input_shape):
#    data_input = Input(shape=input_shape)
#    x = conv_block(data_input, filters=8, kernel_size=3, strides=1, se=False, ratio=8, act='relu', name='block1')  
#    x = conv_block(data_input, filters=8, kernel_size=3, strides=1, se=False, ratio=8, act='relu', name='block2')     
#    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)
#
#    x=ResBlockv2(x, filters=16, kernel_size=3, strides=2, act='relu', ratio=8, name='block31')    
#    x=ResBlockv2(x, filters=16, kernel_size=3, strides=1, act='relu', ratio=8, name='block32')
##    x=ResBlockv2(x, filters=16, kernel_size=3, strides=1, act='relu', ratio=8, name='block33')   
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
def CNN_arch(input_shape):
    data_input = Input(shape=input_shape)
    x = Conv2D(filters=8, kernel_size=3, strides=1,name='block11')(data_input)
    x = Conv2D(filters=8, kernel_size=3, strides=1,name='block12')(data_input)    
    x = MaxPooling2D(pool_size=2, strides=2, padding='same')(x)

    x=ResBlockv2_2D(x, filters=16, kernel_size=3, strides=2, act='relu', ratio=8, name='block21')    
    x=ResBlockv2_2D(x, filters=16, kernel_size=3, strides=1, act='relu', ratio=8, name='block22')
#    x=ResBlockv2_2D(x, filters=16, kernel_size=3, strides=1, act='relu', ratio=8, name='block23')
    
    x = MaxPooling2D(pool_size=2, strides=2, padding='same')(x)
    
    x = GlobalAveragePooling2D()(x) 
    x = Dense(2, kernel_initializer='VarianceScaling', activation='softmax')(x)   
    model = Model(inputs=data_input, outputs=x)
    model.compile(optimizer=Adam(lr=0.001), loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

#=============================================================================#
############################ test function ####################################
#=============================================================================#  
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
###################### test the CNN model: track dondition ####################
#=============================================================================#
FolderPath_train = os.path.abspath(os.path.join(os.getcwd(), "./data/"))
for file in os.listdir(FolderPath_train):
    if file.endswith('.npy'):
        VarName, ext = os.path.splitext(file)
        FilePath = os.path.join(FolderPath_train, file)
        locals()[VarName] = np.load(FilePath)

#=============================================================================#
######################### pick the abnormal data ##############################
#=============================================================================#
# test
indics_bad=np.array([5,42,54,55,74,111,147,357,358,363,376,548,563,571,575,627,650,691,732,756,806,813,
                     833,848,924,954,962,987,990,1191,1203,1256,1281])

indics_peak=np.array([28,38,102,110,123,136,144,154,160,164,171,181,182,196,199,207,237,251,
                      252,266,275,276,279,305,306,311,334,339,344,367,381,389,400,401,412,414,
                      419,423,426,443,465,476,493,499,522,526,530,532,554,581,612,629,642,662,
                      663,687,690,698,704,717,724,727,731,783,789,793,800,834,839,841,
                      858,896,898,900,911,916,917,941,947,948,956,963,967,994,1009,1014,1016,
                      1018,1022,1035,1041,1048,1052,1078,1105,1134,1162,1173,1177,1182,1197,1202,
                      1205,1206,1216,1219,1223,1224,1234,1237,1239,1297,1318,1329])

indics_osc=np.array([167,168,225,227,232,274,340,354,369,432,433,507,523,531,640,652,807,
                     856,866,986,1060,1072,1085,1096,1100,1111,1136,1198,1214,1284,1290])

indics_per=np.array([6,63,70,87,216,255,269,317,335,379,438,501,521,528,559,580,589,596,604,617,636,
                     638,672,681,694,695,719,764,766,862,881,895,927,931,932,1010,1019,1030,1044,1097,
                     1140,1144,1171,1174,1227,1264,1267,1293,1301,1315])    

x_test_AbBad = x_test[indics_bad]
v_test_AbBad = v_test[indics_bad]
y_test_AbBad = y_test[indics_bad]
xraw_test_AbBad = xraw_test[indics_bad]    

x_test_peak = x_test[indics_peak]
v_test_peak = v_test[indics_peak]
y_test_peak = y_test[indics_peak]
xraw_test_peak = xraw_test[indics_peak]  

x_test_osc = x_test[indics_osc]
v_test_osc = v_test[indics_osc]
y_test_osc = y_test[indics_osc]
xraw_test_osc = xraw_test[indics_osc]  

x_test_per = x_test[indics_per]
v_test_per = v_test[indics_per]
y_test_per = y_test[indics_per]
xraw_test_per = xraw_test[indics_per] 

indics_test=np.concatenate((indics_bad,indics_peak,indics_osc,indics_per))
x_test=np.delete(x_test,indics_test,0)
v_test=np.delete(v_test,indics_test,0)
y_test=np.delete(y_test,indics_test,0) 
xraw_test=np.delete(xraw_test,indics_test,0)


# train
indics_bad=np.array([0,6,21,30,44,65,97,98,99,120,132,214,229,245,289,320,329,338,358,375,397,410,425,450,467,492,518,525,529,530,543,603,647,690,
                     692,709,901,918,963,1162,1182,1193,1298,1366,1371,1375,1385,1392,1408,1533,1545,1663,1697,1705,1737,1783,1793,1805,1828,1843,
                     1848,1854,1855,1938,1960,1971,1979,2014,2029,2045,2061,2063,2168,2176,2180,2186,2241,2248,2253,2279,2301,2377,2390,2397,2405,
                     2468,2513,2526,2529,2552,2556,2599,2608,2661,2663,2691,2766,2944,2981,3038,3055,3124,3150,3157,3172,3189,3202,3251,3277,3367,
                     3374,3413,3459,3518,3528,3544,3571,3593,3655,3750,3869,3964,4002])

indics_peak=np.array([20,31,38,41,48,67,77,88,90,102,104,106,107,113,127,151,175,192,200,259,268,278,301,340,352,362,367,376,383,413,419,427,435,
                      436,442,443,459,469,474,475,487,531,538,539,544,547,548,549,557,560,564,578,582,591,599,606,619,620,629,641,648,661,662,664,
                      682,684,686,710,728,752,762,769,781,814,843,852,854,860,868,872,894,900,922,923,971,972,986,992,999,1002,1023,1043,1047,1056,
                      1060,1061,1083,1133,1136,1142,1143,1156,1160,1165,1174,1178,1181,1186,1209,1227,1234,1237,1257,1260,1263,1271,1286,1290,1301,
                      1367,1369,1374,1382,1391,1393,1401,1406,1410,1421,1426,1442,1448,1449,1450,1516,1555,1564,1565,1577,1583,1592,1596,1610,1611,
                      1616,1625,1626,1632,1649,1667,1720,1722,1726,1731,1744,1747,1761,1762,1781,1806,1818,1822,1823,1856,1863,1873,1888,1892,1904,
                      1931,1966,1973,1978,1985,2005,2016,2027,2036,2068,2080,2081,2096,2099,2113,2120,2139,2154,2157,2158,2161,2205,2220,2230,2247,
                      2266,2286,2301,2303,2310,2321,2339,2341,2357,2358,2363,2368,2402,2403,2406,2437,2481,2484,2503,2510,2582,2543,2550,2551,2592,
                      2605,2607,2615,2631,2633,2636,2639,2642,2646,2664,2667,2675,2677,2686,2693,2703,2705,2722,2723,2725,2728,2744,2748,2749,2750,
                      2757,2760,2769,2778,2788,2819,2823,2842,2844,2846,2862,2871,2882,2890,2932,2943,2952,2969,2991,2999,3024,3032,3033,3041,3045,
                      3052,3058,3069,3073,3076,3088,3089,3108,3109,3129,3138,3199,3227,3235,3266,3276,3285,3291,3334,3344,3375,3433,3441,3447,3480,
                      3495,3536,3539,3562,3585,3627,3636,3653,3660,3670,3683,3697,3700,3721,3740,3767,3776,3777,3781,3789,3792,3807,3823,3824,3839,
                      3843,3859,3891,3913,3931,3934,3957,3967,3975,3977,3986,3991,3994,3999])

indics_osc=np.array([70,119,126,128,169,180,299,306,335,420,428,464,508,562,573,644,656,695,707,715,723,729,742,750,760,763,819,828,846,858,865,882,
                     937,977,1001,1009,1024,1034,1053,1075,1092,1094,1103,1112,1131,1147,1155,1241,1285,1302,1305,1340,1523,1525,1537,1567,1618,1619,
                     1620,1628,1648,1672,1677,1681,1710,1723,1735,1738,1746,1763,1812,1850,1875,1970,1993,2072,2093,2102,2112,2130,2155,2166,2213,2225,
                     2227,2229,2295,2335,2345,2366,2453,2485,2563,2645,2660,2737,2771,2876,2913,3030,3180,3241,3253,3290,3393,3449,3516,3590,3833,3848,
                     3854,3873,3919,3936,3952,3962])

indics_per=np.array([9,16,50,57,74,78,143,152,154,161,199,203,250,287,291,294,322,328,404,523,537,542,608,815,897,903,956,1006,1026,1117,1163,1199,
                     1231,1306,1307,1322,1330,1331,1336,1357,1409,1416,1542,1546,1571,1614,1825,1833,1839,1865,1886,1923,1943,2153,2223,2228,2235,2260,
                     2262,2270,2271,2290,2294,2323,2346,2384,2460,2491,2525,2541,2581,2582,2603,2606,2752,2770,2833,2843,2854,2900,2905,2929,2942,2948,
                     2966,3019,3046,3072,3099,3112,3147,3184,3188,3240,3250,3256,3274,3326,3357,3382,3391,3394,3423,3427,3468,3481,3488,3524,3533,3555,
                     3560,3608,3633,3652,3664,3676,3710,3775,3779,3787,3821,3840,3878,3900,3926,3960,3973,3996])  

x_test_AbBad = np.vstack((x_test_AbBad,x_train[indics_bad]))
v_test_AbBad = np.concatenate((v_test_AbBad,v_train[indics_bad]))
y_test_AbBad = np.concatenate((y_test_AbBad,y_train[indics_bad]))
xraw_test_AbBad = np.vstack((xraw_test_AbBad,xraw_train[indics_bad]))

x_test_peak = np.vstack((x_test_peak,x_train[indics_peak]))
v_test_peak = np.concatenate((v_test_peak,v_train[indics_peak]))
y_test_peak = np.concatenate((y_test_peak,y_train[indics_peak]))
xraw_test_peak = np.vstack((xraw_test_peak,xraw_train[indics_peak]))

x_test_osc = np.vstack((x_test_osc,x_train[indics_osc]))
v_test_osc = np.concatenate((v_test_osc,v_train[indics_osc]))
y_test_osc = np.concatenate((y_test_osc,y_train[indics_osc]))
xraw_test_osc = np.vstack((xraw_test_osc,xraw_train[indics_osc]))

x_test_per = np.vstack((x_test_per,x_train[indics_per]))
v_test_per = np.concatenate((v_test_per,v_train[indics_per]))
y_test_per = np.concatenate((y_test_per,y_train[indics_per]))
xraw_test_per = np.vstack((xraw_test_per,xraw_train[indics_per]))

indics_train=np.concatenate((indics_bad,indics_peak,indics_osc,indics_per))
x_train=np.delete(x_train,indics_train,0)
v_train=np.delete(v_train,indics_train,0)
y_train=np.delete(y_train,indics_train,0) 
xraw_train=np.delete(xraw_train,indics_test,0)

# val
indics_bad=np.array([12,20,22,29,41,48,64,113,114,125,138,194,212,223,250,273,284,287,289,298,304,327,331,332,339,347,
                     355,378,388,417,441,444,449,484,491,540,552,557,579,586,588,592,593,598,632,642,663,672,688,719,
                     735,744,760,769,870,883,904,905,976,1031,1048,1064,1090,1098,1100,1128,1213,1219,1248,1287,1305])

indics_peak=np.array([7,10,18,23,35,43,44,51,53,76,77,78,80,82,85,109,120,121,122,127,140,151,15,160,163,177,207,
                      222,226,227,234,252,272,291,330,337,356,357,365,372,380,390,391,404,427,436,443,446,455,
                      459,483,493,497,498,526,538,539,546,573,589,594,605,629,638,648,661,680,683,687,695,689,
                      700,701,703,705,716,723,725,726,728,743,751,754,757,774,788,790,800,809,818,823,858,866,
                      885,891,902,907,932,947,948,953,964,979,992,1005,1018,1027,1044,1054,1061,1063,1065,1099,
                      1101,1109,1115,1116,1119,1122,1129,1137,1149,1151,1152,1165,1171,1186,1199,1201,1202,1218,
                      1224,1233,1236,1240,1252,1264,1267,1273,1275,1277,1291,1292,1293,1309,1338])

indics_osc=np.array([19,33,57, 158,224,230,248,257,361,370,477,507,517,522,554,565,613,633,692,721,730,737,832,841,
                     856,884,1000,1057,1076,1160,1164,1222,1253,1322])

indics_per=np.array([2,17,19,26,104,115,117, 137,171,197, 205,239,267,271,281,283,306,393,408,430,475,476,496,500,536,556,
                     599, 631,634,655,673,707,708,712,729,734,741,777,792,796,833,851,868,876,903,922,951,960,963,966,
                     987,1001,1017,1043,1046,1067,1071,1082,1111,1124,1147,1156,1157,1245,1254,1283,1290,1298,1311,1319,
                     1330])  

x_test_AbBad = np.vstack((x_test_AbBad,x_val[indics_bad]))
v_test_AbBad = np.concatenate((v_test_AbBad,v_val[indics_bad]))
y_test_AbBad = np.concatenate((y_test_AbBad,y_val[indics_bad]))
xraw_test_AbBad = np.vstack((xraw_test_AbBad,xraw_val[indics_bad]))

x_test_peak = np.vstack((x_test_peak,x_val[indics_peak]))
v_test_peak = np.concatenate((v_test_peak,v_val[indics_peak]))
y_test_peak = np.concatenate((y_test_peak,y_val[indics_peak]))
xraw_test_peak = np.vstack((xraw_test_peak,xraw_val[indics_peak]))

x_test_osc = np.vstack((x_test_osc,x_val[indics_osc]))
v_test_osc = np.concatenate((v_test_osc,v_val[indics_osc]))
y_test_osc = np.concatenate((y_test_osc,y_val[indics_osc]))
xraw_test_osc = np.vstack((xraw_test_osc,xraw_val[indics_osc]))

x_test_per = np.vstack((x_test_per,x_val[indics_per]))
v_test_per = np.concatenate((v_test_per,v_val[indics_per]))
y_test_per = np.concatenate((y_test_per,y_val[indics_per]))
xraw_test_per = np.vstack((xraw_test_per,xraw_val[indics_per]))

indics_val=np.concatenate((indics_bad,indics_peak,indics_osc,indics_per))
x_val=np.delete(x_val,indics_val,0)
v_val=np.delete(v_val,indics_val,0)
y_val=np.delete(y_val,indics_val,0) 
xraw_val=np.delete(xraw_val,indics_test,0)

#=============================================================================#
###################### add synthetic data #####################################
#=============================================================================#
#n_syn=500
#x_train_syn=np.zeros((n_syn*6,1000,1))  
#y_train_syn=np.zeros((n_syn*6,2))
#count=0
#FolderPath_syn = os.path.abspath(os.path.join(os.getcwd(), "../data"))
#FolderPath1 = ['v70','v50','v20']
#FolderPath2 = ['WF','normal']
#for i in range(len(FolderPath1)):
#    for j in range(len(FolderPath2)):
#        FileName = 'syn_'+ FolderPath2[j] + '_' + FolderPath1[i] + '.npy'
#        FilePath = os.path.join(FolderPath_syn, FileName)
#        VarName = 'syn_'+ FolderPath2[j] + '_' + FolderPath1[i]
#        globals()[VarName] = np.load(FilePath)                
#        count=count+1
#        x_train_syn[int((count-1)*n_syn):int((count)*n_syn)]=np.reshape(globals()[VarName][:n_syn],(n_syn,1000,1))
#        if FolderPath2[j]=='normal':
#            y_train_syn[int((count-1)*n_syn):int((count)*n_syn),1]=1
#        else:
#            y_train_syn[int((count-1)*n_syn):int((count)*n_syn),0]=1
#                  
#x_train=np.append(x_train,x_train_syn,axis=0)        
#y_train=np.append(y_train,y_train_syn,axis=0)    

#=============================================================================#
###################### pre-processing #########################################
#=============================================================================#
#fs=500
#x_test=ExtraFeatures(x_test,fs)
#csvpath_test= os.path.join(FolderPath_train, 'x_test.csv')
#x_test.to_csv(csvpath_test,index=False) 
#
#x_test_AbBad=ExtraFeatures(x_test_AbBad,fs)
#csvpath_test= os.path.join(FolderPath_train, 'x_test_AbBad.csv')
#x_test_AbBad.to_csv(csvpath_test,index=False) 
#
#x_test_osc=ExtraFeatures(x_test_osc,fs)
#csvpath_test= os.path.join(FolderPath_train, 'x_test_osc.csv')
#x_test_osc.to_csv(csvpath_test,index=False) 
#
#x_test_peak=ExtraFeatures(x_test_peak,fs)
#csvpath_test= os.path.join(FolderPath_train, 'x_test_peak.csv')
#x_test_peak.to_csv(csvpath_test,index=False) 
#
#x_test_per=ExtraFeatures(x_test_per,fs)
#csvpath_test= os.path.join(FolderPath_train, 'x_test_per.csv')
#x_test_per.to_csv(csvpath_test,index=False) 
#
#try:
#    x_train=ExtraFeatures(x_train,fs)
#    csvpath_train= os.path.join(FolderPath_train, 'x_train.csv')
#    x_train.to_csv(csvpath_train,index=False) 
#    
#    x_val=ExtraFeatures(x_val,fs)
#    csvpath_val= os.path.join(FolderPath_train, 'x_val.csv')
#    x_val.to_csv(csvpath_val,index=False) 
#except:
#    pass 

FolderPath_train = os.path.abspath(os.path.join(os.getcwd(), "./data/"))
for file in os.listdir(FolderPath_train):
    if file.endswith('.csv'):
        VarName, ext = os.path.splitext(file)
        FilePath = os.path.join(FolderPath_train, file)  
        locals()[VarName] = pd.read_csv(FilePath)
#
#x_test=np.asarray(x_test)
#x_train=np.asarray(x_train)
#x_val=np.asarray(x_val)
#x_test_AbBad=np.asarray(x_test_AbBad)
#x_test_osc=np.asarray(x_test_osc)
#x_test_peak=np.asarray(x_test_peak)
#x_test_per=np.asarray(x_test_per)

              
# wavelet paket transformation
#level=3
#x_train=wpt(x_train,level)
#x_val=wpt(x_val,level)
#x_test=wpt(x_test,level)
#
#x_test_AbBad=wpt(x_test_AbBad,level)
#x_test_osc=wpt(x_test_osc,level)
#x_test_peak=wpt(x_test_peak,level)
#x_test_per=wpt(x_test_per,level)

### STFT
#fs=500    
#x_train=stft_powerspectrum(x_train,fs)
#x_val=stft_powerspectrum(x_val,fs)
#x_test=stft_powerspectrum(x_test,fs)  
#
#x_test_AbBad=stft_powerspectrum(x_test_AbBad,fs)
#x_test_osc=stft_powerspectrum(x_test_osc,fs)
#x_test_peak=stft_powerspectrum(x_test_peak,fs)
#x_test_per=stft_powerspectrum(x_test_per,fs)

#=============================================================================#
###################### test the CNN model  ####################################
#=============================================================================#
       
# train and test
metrics_train=np.zeros((10,1))
metrics_test_AbBad=np.zeros((10,1))
metrics_test_osc=np.zeros((10,1))
metrics_test_peak=np.zeros((10,1))
metrics_test_per=np.zeros((10,1))


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
#    model=CNN_arch((131,8))
#    model=CNN_arch((33,33,1))
#    model=CNN_arch((1000,1))                                                  # build model
#    history = model.fit(x_train,y_train,batch_size=32,epochs=20,validation_data=[x_val, y_val])  #fiting with hyperparameters
    model = GradientBoostingClassifier()
    model.fit(x_train, y_train[:,1])   
    
    FolderPath_test = os.path.abspath(os.path.join(os.getcwd(), ("./data/")))
    for file in os.listdir(FolderPath_test):
        if file.endswith('.csv'):
            VarName, ext = os.path.splitext(file)
            FilePath = os.path.join(FolderPath_test, file)  
            locals()[VarName] = pd.read_csv(FilePath)   

    acc,_,_,_,_=testing(model,x_test,y_test)
    metrics_train[n,0]=acc
     
    acc,_,_,_,_=testing(model,x_test_AbBad,y_test_AbBad)
    metrics_test_AbBad[n,0]=acc
           
    acc,_,_,_,_=testing(model,x_test_osc,y_test_osc)
    metrics_test_osc[n,0]=acc
        
    acc,_,_,_,_=testing(model,x_test_peak,y_test_peak)
    metrics_test_peak[n,0]=acc
            
    acc,_,_,_,_=testing(model,x_test_per,y_test_per)
    metrics_test_per[n,0]=acc
       
  
#    criteria=np.mean(metrics_train_ESZUEG[n,0])
#    if criteria>best_criteria:
#        best_criteria=criteria
#        model.save('./CNN_AB_v15v105_run'+str(n))
#        np.save('train_acc_best',history.history['acc'])
#        np.save('val_acc_best',history.history['val_acc'])
#        np.save('train_loss_best',history.history['loss'])
#        np.save('val_loss_best',history.history['val_loss'])
        #summarize history for accuracy
