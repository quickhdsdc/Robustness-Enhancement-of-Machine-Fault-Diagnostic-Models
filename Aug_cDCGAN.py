import numpy as np
from numpy import zeros, ones
from numpy.random import randn, randint
from sklearn.preprocessing import StandardScaler
from scipy import signal
import os
import matplotlib.pyplot as plt
from tensorflow.keras import models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, LeakyReLU, BatchNormalization, Embedding, Activation, Concatenate, Conv1D, Conv2DTranspose

# define the standalone discriminator model
def define_discriminator(in_shape=(1024,1), n_classes=2):
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
    out1 = Dense(1, activation='sigmoid')(fe)
    # class label output
    out2 = Dense(n_classes, activation='softmax')(fe)
    # define model
    model = Model(in_image, [out1, out2])
    # compile model
    opt = Adam(lr=0.003, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)
    model.summary()
    return model    
    

# define the standalone generator model
def define_generator(latent_dim, n_classes=2):
    # weight initialization
    #init = RandomNormal(stddev=0.02)
    depth = 16 #32
    dim = 64 #
    # 
    # label input
    in_label = Input(shape=(1,))
    # embedding for categorical input
    li = Embedding(n_classes, 50)(in_label)
    # linear multiplication
    n_nodes = dim * 1
    li = Dense(n_nodes)(li)
    
    # reshape to additional channel
    li = Reshape((dim, 1, 1))(li)
    # image generator input
    in_lat = Input(shape=(latent_dim,))
    # foundation for 7x7 image
    n_nodes = dim*depth
    gen = Dense(n_nodes)(in_lat)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Reshape((dim, 1, depth))(gen)
    # merge image gen and label input
    merge = Concatenate()([gen, li]) #gen=64,1,16 x li=64,1,1
    # upsample to 128,1,16
    gen = Conv2DTranspose(16, 4, strides=(2,1), padding='same')(merge)
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

    #10240 x 1 property image
    gen = Reshape((1024,16))(gen)
    print(gen.shape)
    gen = Conv1D(1, 3, strides=1, padding='same')(gen)
    out_layer = Activation('tanh')(gen)
    # define model
    model = Model([in_lat, in_label], out_layer)
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
    opt = Adam(lr=0.0008, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)
    return model
 
  
def readdata(speeds,dataset,baseline):
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
        
        good.append(good_speed)
        bad.append(bad_speed)
        
    good=np.vstack(good)
    n = len(good)
    label_good = np.reshape(np.zeros(n),(n,1)) # 0 for good
    bad=np.vstack(bad)
    n = len(bad)
    label_bad = np.reshape(np.ones(n),(n,1))  # 1 for bad 
    
    x = np.vstack((good,bad))
    y = np.vstack((label_good,label_bad))
    
    if baseline == 'ResNet':
        [b,a]=signal.butter(5,[256/(5000/2)],'low')
        x_down = np.zeros((len(x),1024))
        for i in range(len(x)):
            each=signal.filtfilt(b, a, x[i,:])
            each=signal.resample(each,1024)
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
    return [x,y]     


# select real samples
def generate_real_samples(dataset, n_samples):
    # split into images and labels
    images, labels = dataset
    # choose random instances
    ix = randint(0, images.shape[0], n_samples)
    # select images and labels
    X, labels = images[ix], labels[ix]
    # generate class labels
    y = ones((n_samples, 1))
    return [X, labels], y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=2):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
    # generate labels
    labels = randint(0, n_classes, n_samples) #check these labels!
    return [z_input, labels]

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
    # generate points in latent space
    z_input, labels_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    images = generator.predict([z_input, labels_input])
    # create class labels
    y = zeros((n_samples, 1))
    return [images, labels_input], y
 
# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, latent_dim, n_samples=100):
    # prepare fake examples
    folderpath = './Results/CDCGAN/'
    [X, nmn_label], nmn_y = generate_fake_samples(g_model, latent_dim, n_samples) #TODO!:Numan (nmns were _ and _) - change labels in this row and debug!
    # plot images
    j = 0
    for i in range(100):
        # np.save(folderpath+'test_raw_nc%d%d.npy' % (i,step), X[i,:])
        if nmn_label[i] == 1:
            j = 1 + j
            if j <= 25:
                plt.subplot(5, 5, j)
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
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=64):
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
        [X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
        # update discriminator model weights
        _,d_r1,d_r2 = d_model.train_on_batch(X_real, [y_real, labels_real])
        # generate 'fake' examples
        [X_fake, labels_fake], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        # update discriminator model weights
        _,d_f,d_f2 = d_model.train_on_batch(X_fake, [y_fake, labels_fake])
        # prepare points in latent space as input for the generator
        [z_input, z_labels] = generate_latent_points(latent_dim, n_batch)
        # create inverted labels for the fake samples
        y_gan = ones((n_batch, 1))
        # update the generator via the discriminator's error
        _,g_1,g_2 = gan_model.train_on_batch([z_input, z_labels], [y_gan, z_labels])
        # summarize loss on this batch
        print('>%d, dr[%.3f,%.3f], df[%.3f,%.3f], g[%.3f,%.3f]' % (i+1, d_r1,d_r2, d_f,d_f2, g_1,g_2))
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
# load image data
dataset = ['ESZUEG']
speeds = ['v35v75_clean']
baselines = ['ResNet','CWT+ResNet','HT+ResNet','GBDT']
data = readdata(speeds,dataset,baselines[0])  
# train model
train(generator, discriminator, gan_model, data, latent_dim)

## generate data
n_syn = 500
# have to manually select a model
g_model = models.load_model('./Results/CDCGAN/'+'model_6302.h5')
[a_syn_v35v75_cdcgan, nmn_label], nmn_y = generate_fake_samples(g_model, latent_dim, n_syn)
a_syn_v35v75_cdcgan = signal.resample(a_syn_v35v75_cdcgan,10240,axis=1)
a_syn_Bad_v35v75_cdcgan = a_syn_v35v75_cdcgan[np.where(nmn_label==1)][:500,:,0]
a_syn_Good_v35v75_cdcgan = a_syn_v35v75_cdcgan[np.where(nmn_label==0)][:500,:,0]


