import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#%matplotlib inline
from pandas import read_csv
import glob
from numpy import dstack
from numpy import vstack
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
import tensorflow as tf
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import to_categorical
from matplotlib import pyplot

def create_model(trainX_acc, trainX_gyro, trainy, testX_acc, testX_gyro, testy):
  n_timesteps_acc, n_features_acc, n_timesteps_gyro, n_features_gyro = trainX_acc.shape[1], trainX_acc.shape[3], trainX_gyro.shape[1], trainX_gyro.shape[3]    
  n_outputs = trainy.shape[1]

  input_acc = Input(shape=(n_timesteps_acc,1,n_features_acc))
  input_gyro = Input(shape=(n_timesteps_gyro,1,n_features_gyro))
  x = Conv2D(16, (10, 1), padding='same', activation='relu')(input_acc)
  x = Reshape((n_timesteps_acc,16))(x)
  #x = np.squeeze(x, axis=(2,))
  #x = LSTM(100,return_sequences=True, input_shape=(n_timesteps_acc,n_features_acc))(input_acc)
  x = LSTM(100,return_sequences=True)(x)
  x = LSTM(100)(x)

  y = Conv2D(16, (10, 1), padding='same', activation='relu')(input_gyro)
  y = Reshape((n_timesteps_gyro,16))(y)
  #y = np.squeeze(y, axis=(2,))
  #y = LSTM(100,return_sequences=True, input_shape=(n_timesteps_gyro,n_features_gyro))(input_gyro)
  y = LSTM(100,return_sequences=True)(y)
  y = LSTM(100)(y)

  combined = concatenate([x, y],axis=1) #axis = 0 or axis = 1?? x and y both have 25 dimension, do we want 25 dimension or 50 dimension?? axis = 1 gives 50 dimension
                                        # axis = 0 gives some kind of error :v

  z = Dense(100,activation='relu')(combined)
  z = Dense(100,activation='relu')(z)
  z = Dense(n_outputs,activation='softmax')(z)

  model = Model(inputs=[input_acc, input_gyro], outputs=z)
  #print(model.summary)
  return model
  
#mymodel = create_model(trainX_acc, trainX_gyro, trainy, testX_acc, testX_gyro, testy)
#print(mymodel.summary())


def evaluate_model(trainX_acc, trainX_gyro, trainy, testX_acc, testX_gyro, testy):
  verbose, epochs, batch_size = 1, 20, 10
  model = create_model(trainX_acc, trainX_gyro, trainy, testX_acc, testX_gyro, testy)
  print(model.summary())
  model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
  history = model.fit([trainX_acc,trainX_gyro], trainy, validation_data=([testX_acc,testX_gyro], testy), epochs=epochs, batch_size=batch_size, verbose=verbose)
  return model


def run_training(repeats=1):
  # load data
  trainX_acc, trainX_gyro, trainy, testX_acc, testX_gyro, testy = load_dataset()
  conv_trainX_acc = np.expand_dims(trainX_acc,axis=2)
  conv_trainX_gyro = np.expand_dims(trainX_gyro,axis=2)
  conv_testX_acc = np.expand_dims(testX_acc,axis=2)
  conv_testX_gyro = np.expand_dims(testX_gyro,axis=2)

  #for r in range(repeats):
  #model = evaluate_model(trainX_acc, trainX_gyro, trainy, testX_acc, testX_gyro, testy)
  model = evaluate_model(conv_trainX_acc, conv_trainX_gyro, trainy, conv_testX_acc, conv_testX_gyro, testy)
  return model

#model = run_training()