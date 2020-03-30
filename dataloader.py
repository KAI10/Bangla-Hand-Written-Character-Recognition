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


def load_file(filepath):
	#dataframe = read_csv(filepath, skiprows = 1, header=None, names=['unix_timestamp', 'sensor_id', 'accuracy', 'x_axis_val', 'y_axis_val', 'z_axis_val'])
  dataframe = read_csv(filepath, skiprows = 1, header=None, names=['x_axis_val', 'y_axis_val', 'z_axis_val'])
  return dataframe.values


def get_max_size():
  mylist = [f for f in glob.glob("data/single/test/*.csv")]
  mylist += [f for f in glob.glob("data/single/train/*.csv")]
  mx = 0
  mn = 800
  for s in mylist:
    print(s)
    cur_data = load_file(s)
    ro = cur_data.shape[0]
    #print(ro)
    mx = max(mx,ro)
    mn = min(mn,ro)
  return mx,mn

#print("mxsize: "+str(mxsize)+" mnsize: "+str(mnsize))


# data = load_file('data/single/G5NZCJ017507206-config.json-ashik-right-v11-single-2020-03-27-12-28-35.csv') 
# print(type(data))
# print(data.shape)



# def load_group(filenames, prefix=''):
# 	loaded = list()
# 	for name in filenames:
# 		data = load_file(prefix + name)
# 		loaded.append(data)
# 	# stack group so that features are the 3rd dimension
# 	loaded = dstack(loaded)
# 	return loaded

def load_file_padded(mxsize, filepath):
  data = load_file(filepath)
  extra = mxsize - data.shape[0]
  for i in range(extra):
    data = np.append(data,[[0.0,0.0,0.0]],axis=0)
  return data

# mylist = [f for f in glob.glob("data/single/train/*.csv")]

# data = load_file_padded(mxsize,mylist[7]) 
# print(type(data))
# print(data.shape)

#print(data)

def get_label(filename):
  pos1 = filename.find("-v")
  pos2 = filename.find("-single")
  #print(str(pos1)+" "+str(pos2))
  val = int(filename[pos1+2:pos2])
  return val

def load_group(filenames, prefix=''): #train or test group.. for train filenames will be from train...
  loaded = list()
  loaded_label = list()
  mxsize,mnsize = get_max_size()
  scaler = MinMaxScaler(feature_range=(-1,1))
  for name in filenames:
    data = load_file_padded(mxsize,prefix + name)
    scaler.fit(data)
    data_normalized = scaler.transform(data)
    #print(name)
    label = get_label(name)
    loaded.append(data_normalized)
    loaded_label.append(label)
  X = np.stack(loaded,axis=0)
  y = np.stack(loaded_label,axis=0)
  y = np.reshape(y,(-1,1))
  return X,y



def load_dataset():
  trainlist = [f for f in glob.glob("data/single/train/*.csv")]
  testlist = [f for f in glob.glob("data/single/test/*.csv")]

  trainX, trainy = load_group(trainlist)
  testX, testy = load_group(testlist)

  print(trainX.shape, trainy.shape) 
  print(testX.shape,testy.shape)
  #print(testy)
  trainy = trainy - 1
  testy = testy - 1
  #print(testy)
  trainy = to_categorical(trainy)
  testy = to_categorical(testy)
  #print(testy)
  print(trainy.shape,testy.shape)


