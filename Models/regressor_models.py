import keras.backend as K
import numpy as np
import os
import sys
import re
import glob
import h5py
import numpy as np
from os.path import join
import setGPU
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Activation, Input, Dense, Dropout, merge, Reshape, Convolution3D, MaxPooling3D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
if __package__ is None:
   sys.path.append(os.path.realpath("/data/shared/RegressionLCD"))
   sys.path.append(os.path.realpath("/data/shared/CMS_Deep_Learning"))
   sys.path.append(os.path.realpath("/nfshome/mliu/LCDstudies/Utils"))
from CMS_Deep_Learning.io import gen_from_data, retrieve_data
import training_utils

def reshapeData(inp):
    (xe, xh), y = inp
    # energy = [y[:,1:]]
    energy = y
    return (xe, xh), energy

def Regressor_DNN():
     #ECAL input
    input1 = Input(shape=(51,51, 25))
    model1 = Flatten()(input1)
     #HCAL input
    input2 = Input(shape=(11, 11, 60))
    model2 = Flatten()(input2)
    # Merging inputs
    bmodel = merge([model1, model2], mode='concat')
    bmodel = (Dense(5, activation='relu'))(bmodel)
    bmodel = (Dropout(0.5))(bmodel)
    oe = Dense(1, activation='linear', name='energy')(bmodel)  # output energy regression
    # energy regression model
    model = Model(input=[input1, input2], output=oe)
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    return model

def Regressor_CNN():
          input1 = Input(shape = (51, 51, 25)) # This returns a tensor
          input2 = Input(shape = (11, 11, 60))
          r = Reshape( (51,51,25,1))(input1)
          model1 = Convolution3D(3, 4, 4, 4, activation='relu')(r)
          model1 = MaxPooling3D()(model1)
          model1 = Flatten()(model1)
          r = Reshape( (11,11,60,1))(input2)
          model2 = Convolution3D(10, 2, 2, 6, activation='relu')(r)
          model2 = MaxPooling3D()(model2)
          model2 = Flatten()(model2)
          # join the two input models
          bmodel = merge([model1,model2], mode='concat') # branched model
          # fully connected ending
          bmodel = (Dense(1000, activation='relu')) (bmodel)
          bmodel = (Dropout(0.5)) (bmodel)
          oe = Dense(1,activation='linear', name='energy')(bmodel) # output energy regression
          emodel = Model(input=[input1,input2], output=oe)
          emodel.compile(loss= 'mse', optimizer='adam')
          return emodel

def main():
    model = Regressor_DNN()
    # defining directory toget the data from:
    data_dir = "/bigdata/shared/LCDLargeWindow/fixedangle/GammaEscan/"
    data_ids = os.listdir(data_dir)
    paths = [glob.glob(join(data_dir, data_id))[0] for data_id in data_ids]
    train_paths = paths[0:3]
    test_paths = paths[3:7]
    valid_paths = paths[7:-1]
    ## Trying the data without the generator first
    # training set
    #train1 = retrieve_data(train_paths[0], data_keys=[["ECAL", "HCAL"], "energy"])
    #(xe,xh), y = train1
    #val1 = retrieve_data(valid_paths[0], data_keys=[["ECAL", "HCAL"], "energy"])
    #(vx,vy), vy = val1
    #hist1 = model.fit([xe, xh], y, nb_epoch=1,
    #                validation_data = val1,
    #                verbose=1)
    #training_utils.saveModel(model, name = "cnntest", outputdir='./tmp')
    train2 = gen_from_data(train_paths, batch_size=500, data_keys=[["ECAL", "HCAL"], "energy"], prep_func=reshapeData)
    # validation set:
    val2 = gen_from_data(valid_paths, batch_size=500, data_keys=[["ECAL", "HCAL"], "energy"], prep_func=reshapeData)
    #testing set:
    test2 = gen_from_data(test_paths, batch_size=500, data_keys=[["ECAL", "HCAL"], "energy"], prep_func=reshapeData)

    hist2 = model.fit_generator(train2, samples_per_epoch=100, 
                           nb_epoch=50,
                           validation_data = val2, 
                           nb_val_samples=100, verbose=1,
                            #validation_split=0.2,
                            #callbacks=[EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min'),
                            #ModelCheckpoint(filepath='simple.h5', verbose=0)]
                            )
if __name__ == "__main__":
    main()
