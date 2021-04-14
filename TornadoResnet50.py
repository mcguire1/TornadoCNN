import numpy as np
import keras
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from resnets_utils import *
from keras.initializers import glorot_uniform
import scipy.misc
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)
import random
import math
import pandas as pd
import os
from netCDF4 import Dataset
from netCDF4 import num2date
import xarray as xr
import time
import dask.array as da
from numpy import isnan
from random import sample
import csv
from skimage import filters
from skimage import exposure
from skimage import segmentation

def ResNet(nc_file_path, tornado_file_path, output_path):
    
    # Read netCDF files from NARR data (example shown for CAPE).
    DS_mf = xr.open_mfdataset(nc_file_path,chunks={'time':5})
    min_x = 64926
    min_y = 64926
    max_x = 10000000
    max_y = 8000000
    mask_lon = (DS_mf.x >= min_x) & (DS_mf.x <= max_x)
    mask_lat = (DS_mf.y >= min_y) & (DS_mf.y <= max_y)
    DS_mf = DS_mf.where(mask_lon & mask_lat, drop=True)
    key_names = list(DS_mf.keys())
    DS_mf_data = DS_mf[key_names[1]]

    #Generate balanced random sample of tornado days for training and testing model
    Y_data = []
    torCounts = pd.read_csv(tornado_file_path)
    for i in range(0,torCounts.shape[0]):
        if (torCounts.iloc[i]["count"]>=0) & (torCounts.iloc[i]["count"]<6):
            Y_data.append([1,0,0,0])
        if (torCounts.iloc[i]["count"]>=6) & (torCounts.iloc[i]["count"]<11):
            Y_data.append([0,1,0,0])
        if (torCounts.iloc[i]["count"]>=11) & (torCounts.iloc[i]["count"]<20):
            Y_data.append([0,0,1,0])
        if (torCounts.iloc[i]["count"]>=20):
            Y_data.append([0,0,0,1])
    Y_data = np.asarray(Y_data)
    a = np.random.choice(np.where(Y_data[:,0]==1)[0],size=500)
    b = np.random.choice(np.where(Y_data[:,1]==1)[0],size=500)
    c = np.random.choice(np.where(Y_data[:,2]==1)[0],size=500)
    d = np.random.choice(np.where(Y_data[:,3]==1)[0],size=500)
    sampleIDX = np.concatenate((a,b,c,d),axis=0)
    sampleIDX = np.sort(sampleIDX)
    Y_data = Y_data[sampleIDX,:]
    
    #Feature Engineering - min-max normalization, exposure equalizastion, and inverse gaussian gradient segmentation
    sampleDS = DS_mf_data.isel(time = sampleIDX)
    data = sampleDS.data
    data = data.compute()
    X_data = data[:,:,:,np.newaxis]
    for i in range(0,X_data.shape[0]):
        for j in range(0,X_data.shape[3]):
            X_data[i,:,:,j] = np.flipud(X_data[i,:,:,j])
            #####################min-max noralization############################
            max_value = np.nanmax(X_data[i,:,:,j])
            min_value = np.nanmin(X_data[i,:,:,j])
            X_data[i,:,:,j] = (X_data[i,:,:,j]-min_value) / (max_value-min_value)
            #####################################################################
            X_data[i,:,:,j] = exposure.equalize_hist(X_data[i,:,:,j])
            X_data[i,:,:,j] = segmentation.inverse_gaussian_gradient(X_data[i,:,:,j]) 
            
    #Partition data for model training and testing
    t = X_data.shape[0]
    sample_index = np.zeros((t,),dtype=int)
    sample_index[random.sample(range(0,t),k=math.floor(t*0.7))]=1
    x_train = X_data[sample_index==1,:,:,:]
    x_test = X_data[sample_index==0,:,:,:]
    y_train = Y_data[sample_index==1]
    y_test = Y_data[sample_index==0]
 
    #Compile and run Resnet model
    model = ResNet50(input_shape=(245, 307, 1), classes=4)
    opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    #Run model with early stopping and learning rate reduction on plateau
    earlyStopping = EarlyStopping(monitor='val_loss', patience=50, verbose=0, mode='min',restore_best_weights=True)
    mcp_save = ModelCheckpoint(output_path+'-{epoch:03d}-{accuracy:03f}-{val_accuracy:03f}.hdf5', save_best_only=True, monitor='val_accuracy', mode='max')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=7, verbose=1, min_lr=1e-10, mode='min')
    model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=100, epochs=500, verbose=1, callbacks=[earlyStopping, mcp_save, reduce_lr_loss])

#Keras API function to create identity block
def identity_block(X, f, filters, stage, block): 
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    #Retrieve Filters
    F1, F2, F3 = filters
    #Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    #Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)
    #Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)
    #Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    return X    

#Keras API function to create convolutional block
def convolutional_block(X, f, filters, stage, block, s = 2):
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    #Retrieve Filters
    F1, F2, F3 = filters
    
    #Save the input value
    X_shortcut = X


    #Main Path
    #First component of main path 
    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    #Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)


    #Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)


    #Shortcut Path
    X_shortcut = Conv2D(filters = F3, kernel_size = (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '1',
                        kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    #Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    
    return X

#Define Resnet 50 model
def ResNet50(input_shape=(245, 307, 1), classes=4):
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    ### START CODE HERE ###

    # Stage 3 (≈4 lines)
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4 (≈6 lines)
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5 (≈3 lines)
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling2D((2,2), name="avg_pool")(X)

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model

if __name__ == "__main__":
    nc_file_path = str(sys.argv[1])
    tornado_file_path = str(sys.argv[2])
    output_path = str(sys.argv[3])
    ResNet(nc_file_path, tornado_file_path, output_path)
