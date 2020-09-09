#=============================================================================#
############################# import functions ################################
#=============================================================================#
from keras.layers import GlobalAveragePooling1D, Multiply, Dense
from keras.models import Model
from keras.layers import Conv1D, Activation, Input, BatchNormalization, Reshape
from keras.layers import MaxPooling1D, SeparableConv1D, Add
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras import backend as K
from keras.optimizers import Adam
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
###################### Build a CNN model: 1DResNet ###########################
#=============================================================================#
#input(10000,1)
def ResNet_10000(input_shape):
    data_input = Input(shape=input_shape)
    x = conv_block(data_input, filters=8, kernel_size=3, strides=2, se=False, ratio=8, act='relu', name='block1')  
    x = conv_block(data_input, filters=8, kernel_size=3, strides=2, se=False, ratio=8, act='relu', name='block2')     
    x = MaxPooling1D(pool_size=10, strides=10, padding='same')(x)

    x=ResBlockv2(x, filters=16, kernel_size=3, strides=2, act='relu', ratio=8, name='block31')    
    x=ResBlockv2(x, filters=16, kernel_size=3, strides=1, act='relu', ratio=8, name='block32')
    x=ResBlockv2(x, filters=16, kernel_size=3, strides=1, act='relu', ratio=8, name='block33')   

    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)
    
    x = GlobalAveragePooling1D()(x) 
    x = Dense(2, kernel_initializer='VarianceScaling', activation='softmax')(x)   
    model = Model(inputs=data_input, outputs=x)
    model.compile(optimizer=Adam(lr=0.001), loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

#=============================================================================#
###################### Build a CNN model: 1DResNet ###########################
#=============================================================================#
#input(1000,1)
def ResNet(input_shape):
    data_input = Input(shape=input_shape)
    x = conv_block(data_input, filters=8, kernel_size=3, strides=2, se=False, ratio=8, act='relu', name='block1')  
    x = conv_block(data_input, filters=8, kernel_size=3, strides=2, se=False, ratio=8, act='relu', name='block2')     
    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)

    x=ResBlockv2(x, filters=16, kernel_size=3, strides=2, act='relu', ratio=8, name='block31')    
    x=ResBlockv2(x, filters=16, kernel_size=3, strides=1, act='relu', ratio=8, name='block32')
     
    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)
    
    x = GlobalAveragePooling1D()(x) 
    x = Dense(2, kernel_initializer='VarianceScaling', activation='softmax')(x)   
    model = Model(inputs=data_input, outputs=x)
    model.compile(optimizer=Adam(lr=0.001), loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model
#=============================================================================#
################### Build a CNN model: WPT + 1DResNet #########################
#=============================================================================#
def ResNet_WPT(input_shape):
    data_input = Input(shape=input_shape)
    x = conv_block(data_input, filters=8, kernel_size=3, strides=1, se=False, ratio=8, act='relu', name='block1')  
    x = conv_block(data_input, filters=8, kernel_size=3, strides=1, se=False, ratio=8, act='relu', name='block2')     
    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)

    x=ResBlockv2(x, filters=16, kernel_size=3, strides=2, act='relu', ratio=8, name='block31')    
    x=ResBlockv2(x, filters=16, kernel_size=3, strides=1, act='relu', ratio=8, name='block32')
     
    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)
    
    x = GlobalAveragePooling1D()(x) 
    x = Dense(2, kernel_initializer='VarianceScaling', activation='softmax')(x)   
    model = Model(inputs=data_input, outputs=x)
    model.compile(optimizer=Adam(lr=0.001), loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model
#=============================================================================#
################ Build a CNN model: STFT + 2DResNet ###########################
#=============================================================================#
def ResNet_STFT(input_shape):
    data_input = Input(shape=input_shape)
    x = Conv2D(filters=8, kernel_size=3, strides=1,name='block11')(data_input)
    x = Conv2D(filters=8, kernel_size=3, strides=1,name='block12')(data_input)    
    x = MaxPooling2D(pool_size=2, strides=2, padding='same')(x)

    x=ResBlockv2_2D(x, filters=16, kernel_size=3, strides=2, act='relu', ratio=8, name='block21')    
    x=ResBlockv2_2D(x, filters=16, kernel_size=3, strides=1, act='relu', ratio=8, name='block22')
    
    x = MaxPooling2D(pool_size=2, strides=2, padding='same')(x)
    
    x = GlobalAveragePooling2D()(x) 
    x = Dense(2, kernel_initializer='VarianceScaling', activation='softmax')(x)   
    model = Model(inputs=data_input, outputs=x)
    model.compile(optimizer=Adam(lr=0.001), loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model
