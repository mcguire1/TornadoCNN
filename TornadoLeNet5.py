#!/usr/bin/env python3
#Import necessary Python libraries
import os
import sys
from sys import exit
from netCDF4 import Dataset
from netCDF4 import num2date
import numpy as np
import keras
import pandas as pd
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Activation, Conv2D, AveragePooling2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Dropout
from keras import regularizers
from keras.metrics import categorical_accuracy
from keras.models import load_model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.layers.normalization import BatchNormalization
from sklearn.preprocessing import minmax_scale
from keras.callbacks import CSVLogger
import random
import math
import xarray as xr
import time
import dask.array as da
from numpy import isnan
from random import sample
import csv
from skimage import filters
from skimage import exposure
from skimage import segmentation

def LeNet5(nc_file_path, tornado_file_path, output_path):
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
    
#LeNet-5 Model
    model = Sequential()
    # Layer 1:
    model.add(Conv2D(filters=6, kernel_size=5, strides=1, padding='Same', 
                     input_shape= x_train.shape[1:]))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    # Layer 2:
    model.add(Conv2D(filters=16, kernel_size=5, strides=1))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    # Layer 3:
    model.add(Flatten())
    model.add(Dense(120))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    # Layer 4:
    model.add(Dense(84))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    # Output layer 5:
    model.add(Dense(4))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    
#Compile model using accuracy to measure model performance
    opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy']) 
    
#Run model with early stopping and learning rate reduction on plateau
    earlyStopping = EarlyStopping(monitor='val_loss', patience=50, verbose=0, mode='min',restore_best_weights=True)
    mcp_save = ModelCheckpoint(output_path+'-{epoch:03d}-{accuracy:03f}-{val_accuracy:03f}.hdf5', save_best_only=True, monitor='val_accuracy', mode='max')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=7, verbose=1, min_lr=1e-10, mode='min')
    model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=100, epochs=500, verbose=1, callbacks=[earlyStopping, mcp_save, reduce_lr_loss])

if __name__ == "__main__":
    nc_file_path = str(sys.argv[1])
    tornado_file_path = str(sys.argv[2])
    output_path = str(sys.argv[3])
    LeNet5(nc_file_path, tornado_file_path, output_path)

