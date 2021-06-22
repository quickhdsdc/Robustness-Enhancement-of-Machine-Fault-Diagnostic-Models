import numpy as np
from numpy import zeros, ones
from numpy.random import randn, randint
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import signal
import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, LeakyReLU, BatchNormalization, Embedding, Activation, Concatenate, Conv1D, Conv2DTranspose
from Utils.utils_aug_ft import readdata,segment_signal

# define the standalone discriminator model
def define_discriminator(in_shape=(1024,1)):
    # weight initialization
    #init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=in_shape)
    # block1
    fe = Conv1D(16, 7, strides=2, padding='same')(in_image)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)    
    # block2
    fe = Conv1D(16, 7, strides=2, padding='same')(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    # block3
    fe = Conv1D(16, 5, strides=2, padding='same')(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    # block4
    fe = Conv1D(16, 5, strides=2, padding='same')(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)   
    
    fe = Flatten()(fe)
    # fe = GlobalAveragePooling1D()(fe)     
    # real/fake output
    out = Dense(1, activation='sigmoid')(fe) 
    # define model
    model = Model(in_image, out)
    # compile model
    opt = Adam(lr=0.001, beta_1=0.5)
    model.compile(loss=['binary_crossentropy'], optimizer=opt)
    model.summary()
    return model    
    

# define the standalone generator model
def define_generator(latent_dim):
    # weight initialization
    #init = RandomNormal(stddev=0.02)
    depth = 16 #32
    dropout = 0.25
    dim = 64 #
    # 
    # image generator input
    in_lat = Input(shape=(latent_dim,))
    # foundation for 7x7 image
    n_nodes = dim*depth
    gen = Dense(n_nodes)(in_lat)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Reshape((dim, 1, depth))(gen)
    # upsample to 128,1,16
    gen = Conv2DTranspose(16, 4, strides=(2,1), padding='same')(gen)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    # upsample to 256,1,16
    gen = Conv2DTranspose(16, 4, strides=(2,1), padding='same')(gen)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)    
    #upsample to  512,1,8
    gen = Conv2DTranspose(16, 4, strides=(2,1), padding='same')(gen)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    #upsample to  1024,1,8
    gen = Conv2DTranspose(16, 4, strides=(2,1), padding='same')(gen)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    #1024 x 1 property image
    gen = Reshape((1024,16))(gen)
    print(gen.shape)
    gen = Conv1D(1, 3, strides=1, padding='same')(gen)
    out_layer = Activation('tanh')(gen)
    # define model
    model = Model(in_lat, out_layer)
    model.summary()
    return model
 
# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # connect the outputs of the generator to the inputs of the discriminator
    gan_output = d_model(g_model.output)
    # define gan model as taking noise and label and outputting real/fake and label outputs
    model = Model(g_model.input, gan_output)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy'], optimizer=opt)
    return model
 
  
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
        if health == 'Good':
            good.append(good_speed)
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

    if baseline == 'ResNet':
        x_down = np.zeros((len(x),1024))
        for i in range(len(x)):
            each=signal.resample(x[i],1024)
            # schale to the range [-1, 1]
            x_down_min = np.min(each)
            x_down_max = np.max(each)
            x_down[i,:] = (each - x_down_min)*2/(x_down_max-x_down_min)-1            
        x = x_down
        x = x - np.mean(x, axis=1, keepdims=True)
    elif baseline == 'CWT+ResNet':
        x=signal.resample(x,1024,axis=1) 
    
    elif baseline == 'GBDT':
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        
    indices = list(range(len(x)))  
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices] 
    x = x.reshape((x.shape[0],x.shape[1],1))           
    return x,y   


# select real samples
def generate_real_samples(dataset, n_samples):
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # select images and labels
    X = dataset[ix]
    # generate class labels
    y = ones((n_samples, 1))
    return X, y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
    return z_input

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
    # generate points in latent space
    z_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    data_fake = generator.predict(z_input)
    # create class labels
    y = zeros((n_samples, 1))
    return data_fake, y
 
# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, latent_dim, n_samples=100):
    # prepare fake examples
    folderpath = './Results/SIM-GAN/'
    X, nmn_y = generate_fake_samples(g_model, latent_dim, n_samples) #TODO!:Numan (nmns were _ and _) - change labels in this row and debug!
    # plot images
    for i in range(16):
        plt.subplot(4, 4, i+1)
        # turn off axis
        plt.axis('off')
        # plot raw pixel data
        plt.plot(X[i,:,0])
    
    filename1 = 'generated_plot_%04d.png' % (step+1)
    plt.savefig(folderpath + filename1)
    plt.close()
    # save the generator model
    filename2 = 'model_%04d.h5' % (step+1)
    g_model.save(folderpath + filename2)
    print('>Saved: %s and %s' % (filename1, filename2))
 
# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=150, n_batch=64):
    # calculate the number of batches per training epoch
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    print('batch per epoch: %d' % bat_per_epo)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    print('number of steps: %d' % n_steps)
    # calculate the size of half a batch of samples
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_steps):
        # get randomly selected 'real' samples
        X_real, y_real = generate_real_samples(dataset, half_batch)
        # update discriminator model weights
        d_r = d_model.train_on_batch(X_real, y_real)
        # generate 'fake' examples
        X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        # update discriminator model weights
        d_f = d_model.train_on_batch(X_fake, y_fake)
        # prepare points in latent space as input for the generator
        z_input = generate_latent_points(latent_dim, n_batch)
        # create inverted labels for the fake samples
        y_gan = ones((n_batch, 1))
        # update the generator via the discriminator's error
        g = gan_model.train_on_batch(z_input, y_gan)
        # summarize loss on this batch
        print('>%d, dr[%.3f], df[%.3f], g[%.3f]' % (i+1, d_r, d_f, g))
        # evaluate the model performance every 'epoch'
        if (i+1) % (bat_per_epo * 1) == 0:
            summarize_performance(i, g_model, latent_dim)

# size of the latent space
latent_dim = 100
# create the discriminator
discriminator = define_discriminator()
# create the generator
generator = define_generator(latent_dim)
# create the gan
gan_model = define_gan(generator, discriminator)

####### load data
dataset = ['ESZUEG']
speeds = ['v15v25','v25v35','v35v45']
baselines = ['ResNet','CWT+ResNet','HT+ResNet','GBDT']
x,_ = readdata(speeds,'Bad',dataset,baselines[0])  
##
FolderPath = os.path.abspath(os.path.join(os.getcwd(), "./Data/Augmentation/input_data/"))
sim_normal=pd.read_csv(FolderPath+'/sim_norm_30mm.txt',engine='python')
sim_WF=pd.read_csv(FolderPath+'/sim_WF_30mm.txt',engine='python')                
# time shifting for more data samples     
FolderPath1 = ['v20','v30','v40']
FolderPath2 = ['normal','WF']
for i in range(len(FolderPath1)):
    for j in range(len(FolderPath2)):
        a_varname = 'sim_'+ FolderPath2[j] + '_' +  FolderPath1[i]
        df_varname = 'sim_'+ FolderPath2[j]
        globals()[a_varname]=segment_signal(globals()[df_varname],FolderPath1[i],10240)
# flipping for doubling data samples              
for i in range(len(FolderPath1)):
    for j in range(len(FolderPath2)):
        a_varname = 'sim_'+ FolderPath2[j] + '_' +  FolderPath1[i]      
        n_sample = len(globals()[a_varname])
        temp=np.copy(globals()[a_varname])
        globals()[a_varname]=np.append(globals()[a_varname],temp*-1,axis=0)
        scaler = MinMaxScaler()
        globals()[a_varname]=scaler.fit_transform(globals()[a_varname].T).T
        globals()[a_varname] = globals()[a_varname] - np.mean(globals()[a_varname],axis=1, keepdims=True)
x_sim = np.vstack((sim_WF_v20,sim_WF_v30,sim_WF_v40))
x_down = np.zeros((len(x_sim),1024))
for i in range(len(x_sim)):
    each=signal.resample(x_sim[i],1024)
    # schale to the range [-1, 1]
    x_down_min = np.min(each)
    x_down_max = np.max(each)
    x_down[i,:] = (each - x_down_min)*2/(x_down_max-x_down_min)-1 
x_sim = x_down
x_sim = x_sim - np.mean(x_sim, axis=1, keepdims=True)
x_sim = np.reshape(x_sim,(len(x_sim),1024,1))
x = np.vstack((x,x_sim))
indices = list(range(len(x)))  
np.random.shuffle(indices)
x = x[indices]

# train model
train(generator, discriminator, gan_model, x, latent_dim)

## test
n_syn = 500
# have to manually select a model
g_model = models.load_model('./Results/SIM-GAN/'+'model_1296.h5')
a_syn_Bad_v35v75_simgan, nmn_y = generate_fake_samples(g_model, latent_dim, n_syn)
a_syn_Bad_v35v75_simgan = signal.resample(a_syn_Bad_v35v75_simgan,10240,axis=1)
a_syn_Bad_v35v75_simgan = a_syn_Bad_v35v75_simgan.reshape((n_syn,10240))

