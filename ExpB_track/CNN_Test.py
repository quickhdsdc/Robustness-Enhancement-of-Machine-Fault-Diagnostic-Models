#=============================================================================#
############################# import functions ################################
#=============================================================================#
import random
from keras import models
from keras.models import load_model
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from sklearn.metrics import roc_curve
from sklearn import metrics
from FeatureExtraction import envelope_powerspectrum, fft_powerspectrum
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
import numpy as np
import copy
import math
import seaborn as sns
from sklearn import preprocessing
import tensorflow as tf
import keras.backend as K
from keras.utils import plot_model
from Mish import Mish, Relu6, Hswish
import os
import matplotlib.gridspec as gridspec

#=============================================================================#
############################# load model  #####################################
#=============================================================================#
del model
K.clear_session
model = load_model('./CNN_AB_v15v55')
#=============================================================================#
##########################  model detals  #####################################
#=============================================================================#
def get_flops(model):
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = tf.profiler.profile(graph=K.get_session().graph,
                                run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops  # Prints the "flops" of the model.

# .... Define your model here ....
MAC=get_flops(model)/2
print('MAC = ' + str(MAC))
print('FLOPs = ' + str(get_flops(model)))
model.summary()
plot_model(model, to_file='./CNN_CB_raw_LightWFNet.png', show_layer_names=True, show_shapes=True)
#=============================================================================#
############################# testing function  ###############################
#=============================================================================#
def testing(model,x_test,y_test):  
#    test_results = model.evaluate(x_test, y_test)
#    for name, value in zip(model.metrics_names, test_results):
#        print(name, value)  
    
    y_pred = model.predict(x_test)  
#    y_pred = model.predict_proba(x_test)    
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

##tensorboard --logdir "D:/MachineLearning/WheelFlat/logs/lr_1e-04_batch_35_epoch_10_tanh_f12_f1n27_f256_f2n82_f323_f3n33"
#=============================================================================#
################################# plot functions ##############################
#=============================================================================#
# plot for time-frequency spectrum (wavelet scalegram)[255,5000]
#def plot_data(rawdata,traindata,v,cls_true,cls_pre,indics):
#    assert rawdata.shape[0] == cls_true.shape[0]
#    i = random.choice(indics)
#    if cls_true[i]!=cls_pre[i]:
#        if cls_true[i]:
#            s = 'bad'
#            s1 = 'good'
#        else:
#            s = 'good'
#            s1 = 'bad'            
#    else:
#        if cls_true[i]:
#            s = 'bad'
#            s1 = 'bad'
#        else:
#            s = 'good'
#            s1 = 'good'           
#    fig, axContour = plt.subplots(figsize=(10, 8))
#    traindata=np.reshape(traindata,(len(traindata),32,96))      
#    axContour.pcolormesh(traindata[i])
#    axContour.grid(True)
#    axContour.set_title(str(i)+'th Morlet scalogram v='+str(np.round(v[i]))+ 'km/h  cls_true='+s+' cls_pre='+s1)
#    axContour.set_ylabel('Frequency')
#    axContour.set_xlabel('Time')
#    
#    time = np.linspace(0,1,num=rawdata.shape[1],endpoint=False) 
#    divider = make_axes_locatable(axContour)
#    axTime = divider.append_axes("top", 2, pad=0.8)
#    axTime.plot(time,rawdata[i])
#    axTime.set_ylabel('Acceleration')
#    axTime.set_title(str(i)+'th raw acceleration data v=' + str(np.round(v[i])) + '  cls_true='+s+' cls_pre='+s1)
#    axTime.grid(True)
#    plt.show()

#=============================================================================#
def plot_data(rawdata,traindata,v,cls_true,cls_pre,indics):
    assert rawdata.shape[0] == cls_true.shape[0]
    i = random.choice(indics)
    if cls_true[i]!=cls_pre[i]:
        if cls_true[i]:
            s = 'bad'
            s1 = 'good'
        else:
            s = 'good'
            s1 = 'bad'            
    else:
        if cls_true[i]:
            s = 'bad'
            s1 = 'bad'
        else:
            s = 'good'
            s1 = 'good'           
            
    time = np.linspace(0,1,num=traindata.shape[1],endpoint=False) 
    envelope,Freq = envelope_powerspectrum(rawdata[i],5000,200)       
    fig, (ax0, ax1) = plt.subplots(nrows = 2, figsize = (15, 10), sharex = False)
    plot_axis(ax0, time, traindata[i],str(i)+'th Acceleration v='+str(np.round(v[i]))+ 'km/h  cls_true='+s+' cls_pre='+s1)   
    plot_axis(ax1, Freq, envelope,str(i)+'th Envelope spectrum v=' + str(np.round(v[i])) + '  cls_true='+s+' cls_pre='+s1)   
    plt.subplots_adjust(hspace=0.2)
    plt.subplots_adjust(top=0.90)
    plt.show()

    
def plot_axis(ax, x, y, title):
    ax.plot(x, y)
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)   
#=============================================================================#
# plot function for confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    
    'This function prints and plots the confusion matrix.Normalization can be applied by setting `normalize=True`.'
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax        
#=============================================================================#
################################## plot  data #################################
#=============================================================================#
# plot raw data for wrong classification
index_dataset=2
FolderPath = ['train_v15v75','test_v75v85','test_v85v95','test_v95v105']
FolderPath_test = os.path.abspath(os.path.join(os.getcwd(), ("./ESZUEG/"+FolderPath[index_dataset])))
for file in os.listdir(FolderPath_test):
    VarName, ext = os.path.splitext(file)
    FilePath = os.path.join(FolderPath_test, file)
    locals()[VarName] = np.load(FilePath)
    
#acc,recall,precision,auc,cls_pre,cls_true,indics_false,indics_true=testing(model,x_test,y_test) 
acc,cls_pre,cls_true,indics_false,indics_true=testing(model,x_test_per,y_test_per)    
   
plot_data(xraw_test_per,x_test_per,v_test_per,cls_true,cls_pre,indics_false)
plot_data(xraw_test,x_test,v_test,cls_true,cls_pre,indics_true)

# plot training history
#plt.plot(train_acc_spconvse,'r')
#plt.plot(train_acc_spconv,'b')
#plt.plot(train_acc_baseline,'k')
#plt.plot(val_acc_spconvse,'r--')
#plt.plot(val_acc_spconv,'b--')
#plt.plot(val_acc_baseline,'k--')
#
#plt.title('Training accuracy and validation loss')
#plt.ylabel('accuracy/loss')
#plt.xlabel('epoch')
#plt.legend(['train_acc_spconv+se', 'train_acc_spconv','train_acc_baseline','val_acc_spconv+se','val_acc_spconv','val_acc_baseline'], loc='lower right')
#plt.show()

#=============================================================================#
# plot confusion matrix
np.set_printoptions(precision=2)
class_names = ['good wheel','wheel flat']
class_names=np.asarray(class_names)
# Plot non-normalized confusion matrix
plot_confusion_matrix(cls_true, cls_pre, classes=class_names, normalize=True,
                      title='Normalized Confusion matrix')
plt.show()

#=============================================================================#
##################### Activation Visualization ################################
#=============================================================================#
def plt_act(fig,grid,data,layer_name,layer_activation_index,images_per_row):
    layer_names = []
    for layer in model.layers:
        layer_names.append(layer.name) 
    act_index=[position for position, name in enumerate(layer_names) if layer_name in name]  # index of activation layers
    act_index=np.asarray(act_index)
    
#    actlayer_names=[layer_names[i] for i in act_index]
    layer_outputs =[model.layers[i].output for i in act_index]                      # output of activation layers   
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)      # Creates a model that will return these outputs, given the model input
    test_input = np.expand_dims(data, axis=0)
    activations = activation_model.predict(test_input)
    layer_activation = activations[layer_activation_index]     
    images_per_row = images_per_row
    col=[]
    row=[]  
    n_features = layer_activation.shape[-1] # Number of features in the feature map
    n_rows = math.ceil(n_features/images_per_row) # Tiles the activation channels in this matrix
    inner = gridspec.GridSpecFromSubplotSpec(n_rows, images_per_row,
                    subplot_spec=grid, wspace=0.1, hspace=0.1)      
    if n_rows>1:
        for row in range(n_rows):
            for col in range(images_per_row):
                if row*images_per_row+col < n_features:
                    ax = plt.Subplot(fig, inner[row,col])
                    ax.plot(layer_activation[0,:,(row*images_per_row+col)], color='darkblue')
#                    axes[row,col].plot(layer_activation[0,:,(row*images_per_row+col)], color='darkblue')
                    ax.yaxis.set_visible(False)
                    ax.xaxis.set_visible(False)
                    fig.add_subplot(ax)
                else:
                    ax.yaxis.set_visible(False)
                    ax.xaxis.set_visible(False)
                    fig.add_subplot(ax)
    else:
        for col in range(images_per_row):
            if col < n_features:
                ax = plt.Subplot(fig, inner[col])
                ax.plot(layer_activation[0,:,col], color='darkblue')
                ax.yaxis.set_visible(False)
                ax.xaxis.set_visible(False)
                fig.add_subplot(ax)
            else:
                ax.yaxis.set_visible(False)
                ax.xaxis.set_visible(False) 
                fig.add_subplot(ax)
    return
                               
###############################################################################
fs=5000
index_inputdata = 65
data = x_test[index_inputdata]
t = np.linspace(0, 2, len(data))
###############################################################################
fig = plt.figure(figsize=(10, 8),constrained_layout=True)
outer = gridspec.GridSpec(2, 2, wspace=0.2, hspace=0.2, figure=fig)

ax1 = plt.Subplot(fig, outer[0])
ax1.plot(t,data, color='darkblue')
ax1.yaxis.set_visible(False)
ax1.xaxis.set_visible(False)
#ax1.title.set_text('Raw data')
fig.add_subplot(ax1)

plt_act(fig,outer[0,1],data, 'max_pooling', 0, 4)
plt_act(fig,outer[1,0],data, '_bn',-1, 8)
plt_act(fig,outer[1,1],data, 'max_pooling',1, 8)
fig.show()

fig.savefig('./Figure/LayerOutputs.png',dpi=600)
fig.savefig('./Figure/LayerOutputs.eps',format='eps',dpi=600)

#=============================================================================#
##################### Activation Visualization nonfunction ####################
#=============================================================================#
model.summary()
layer_names = []
for layer in model.layers:
    layer_names.append(layer.name) 
act_index=[position for position, name in enumerate(layer_names) if 'max_pooling1d' in name]  # index of activation layers
act_index=np.asarray(act_index)

actlayer_names=[layer_names[i] for i in act_index]
layer_outputs =[model.layers[i].output for i in act_index]                      # output of activation layers

activation_model = models.Model(inputs=model.input, outputs=layer_outputs)      # Creates a model that will return these outputs, given the model input
test_input = np.expand_dims(data, axis=0)
activations = activation_model.predict(test_input)                              # Returns a list of Numpy arrays: one array per layer activation

plt.plot(data)
images_per_row = 9
col=[]
row=[]
for layer_name,layer_activation in zip(actlayer_names,activations):
    n_features = layer_activation.shape[-1] # Number of features in the feature map
    size = layer_activation.shape[1] #The feature map has shape (1, size, n_features).
    n_rows = math.ceil(n_features/images_per_row) # Tiles the activation channels in this matrix
    fig,axes=plt.subplots(nrows=n_rows, ncols=images_per_row)
    fig.suptitle(layer_name)
    if n_rows>1:
        for row in range(n_rows):
            for col in range(images_per_row):
                if row*images_per_row+col < n_features:
                    axes[row,col].plot(layer_activation[0,:,(row*images_per_row+col)])
                    axes[row,col].yaxis.set_visible(False)
                    axes[row,col].xaxis.set_visible(False)
                else:
                    axes[row,col].yaxis.set_visible(False)
                    axes[row,col].xaxis.set_visible(False)
    else:
        for col in range(images_per_row):
            if col < n_features:
                axes[col].plot(layer_activation[0,:,col])
                axes[col].yaxis.set_visible(False)
                axes[col].xaxis.set_visible(False)
            else:
                axes[col].yaxis.set_visible(False)
                axes[col].xaxis.set_visible(False)  
#=============================================================================#
######################## Filter Visualization #################################
#=============================================================================#
def generate_pattern(layer_name, filter_index, size=500):
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, filter_index])
    grads = K.gradients(loss, model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([model.input], [loss, grads])
    input_data = np.random.random((1, size, 1)) * 5
    step = 1
    for i in range(40):
        loss_value, grads_value = iterate([input_data])
        input_data += grads_value * step
    img = input_data[0]
    return img
##=============================================================================#
for layer in model.layers:
    layer_names.append(layer.name) 
conv_index=[position for position, name in enumerate(layer_names) if 'dense_9' in name]  # index of conv layers
Weights_conv =[model.layers[i].get_weights() for i in conv_index]                      # output of activation layers
convlayer_names=[layer_names[i] for i in conv_index]
images_per_row = 3
n = 6

for layer_name,layer_weights in zip(convlayer_names,Weights_conv):
    layer_weights=np.asarray(layer_weights[0])
    n_features = layer_weights.shape[-1] # Number of features in the feature map
    size = layer_weights.shape[0] #The feature map has shape (1, size, n_features).
    fig,axes=plt.subplots(nrows=2, ncols=images_per_row)
    fig.suptitle(layer_name)
    for col in range(images_per_row):
        if col+n*images_per_row < n_features:
            filter_index=col+n*images_per_row
            output=generate_pattern(layer_name,filter_index=filter_index)
            axes[0,col].plot(output)
            axes[0,col].title.set_text(str(filter_index)+'th filter')
            axes[0,col].yaxis.set_visible(False)
            axes[0,col].xaxis.set_visible(False)
            envelope,Freq = fft_powerspectrum(output,500,200)
            axes[1,col].plot(Freq,envelope)
            axes[1,col].title.set_text(str(filter_index)+'th filter envelope spectrum' )
#            axes[1,col].yaxis.set_visible(False)
#            axes[1,col].xaxis.set_visible(False)                
        else:
            axes[row,col].yaxis.set_visible(False)
            axes[row,col].xaxis.set_visible(False)

#=============================================================================#
######################## Occlusion Sensitivity ################################
#=============================================================================#
min_max_scaler = preprocessing.MinMaxScaler()  
def Occlusion_exp(data, occluding_size, occluding_pixel, occluding_stride):
    test_input = np.expand_dims(data, axis=0)   
    out = model.predict(test_input)
    out = out[0,1]
    width, _ = data.shape
    output_width = int(math.ceil((width-occluding_size) / occluding_stride + 1))
    heatmap = np.zeros((1,output_width))
    
    for w in range(output_width):
        # Occluder region:
        w_start = w * occluding_stride
        w_end = min(width, w_start + occluding_size)
        # Getting the image copy, applying the occluding window and classifying it again:
        input_data = copy.copy(data)
        input_data[w_start:w_end,:] =  occluding_pixel            
        input_data1 = np.expand_dims(input_data, axis=0)   
        out = model.predict(input_data1)
        out = out[0,1]
        print('scanning position (%s)'%(w))
        # It's possible to evaluate the VGG-16 sensitivity to a specific object.
        # To do so, you have to change the variable "index_object" by the index of
        # the class of interest. The VGG-16 output indices can be found here:
        # https://github.com/HoldenCaulfieldRye/caffe/blob/master/data/ilsvrc12/synset_words.txt
        heatmap[0,w] = out
    
#    fig,(ax1,ax2)=plt.subplots(nrows=2, ncols=1)
#    ax1.plot(data)
#    ax2 = sns.heatmap(heatmap,xticklabels=False, yticklabels=False)
    ax=plt.plot(min_max_scaler.fit_transform(data))
    ax = sns.heatmap(heatmap,xticklabels=False, yticklabels=False)
    return heatmap
#=============================================================================#
occluding_size = 200
occluding_pixel = 0
occluding_stride = 1
 
heatmap=Occlusion_exp(data, occluding_size, occluding_pixel, occluding_stride)

max_index=np.argmax(heatmap[0,:])
input_data = copy.copy(data)
input_data[max_index*occluding_stride:min(np.shape(data)[0], max_index*occluding_stride + occluding_size),:] =  occluding_pixel
input_data1 = np.expand_dims(input_data, axis=0)   

prob_original=model.predict(np.expand_dims(data, axis=0))
prob_original=prob_original[0,1]
prob_mod=model.predict(input_data1)
prob_mod=prob_mod[0,1]

fig,(ax1,ax2)=plt.subplots(nrows=2, ncols=1)
ax1.plot(data)
ax1.set_title('original data, probobility as wheel flat: ' + str(round(prob_original*100,2))+'%')
ax2.plot(input_data1[0,:,0])
ax2.set_title('original data, probobility as wheel flat: ' + str(round(prob_mod*100,2))+'%')

