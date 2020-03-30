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

def evaluate_model(trainX, trainy, testX, testy):
  verbose, epochs, batch_size = 1, 20, 20
  n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
  model = Sequential()
  model.add(LSTM(25, return_sequences=True, input_shape=(n_timesteps,n_features)))
	#model.add(Dropout(0.5))
  model.add(LSTM(25))
  model.add(Dense(25, activation='relu'))
  model.add(Dense(25, activation='relu'))
  model.add(Dense(n_outputs, activation='softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit network
  model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
	# evaluate model
  _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
  return model, accuracy

def summarize_results(scores):
	print(scores)
	m, s = mean(scores), std(scores)
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

def run_training(repeats=1):
	# load data
  trainX, trainy, testX, testy = load_dataset()

  #scaler.fit(trainX)
  #trainX_normalized = scaler.transform(trainX)
  #scaler.fit(testX)
  #trainX_normalized = scaler.transform(testX)
  #repeat experiment
  scores = list()
  #for r in range(repeats):
  model, score = evaluate_model(trainX, trainy, testX, testy)
  score = score * 100.0
  #print('>#%d: %.3f' % (r+1, score))
  scores.append(score)
	# summarize results
  summarize_results(scores)
  return model

#model = run_training()

