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

def load_file_seperate(filepath):
  #dataframe = read_csv(filepath, skiprows = 1, header=None, names=['unix_timestamp', 'sensor_id', 'accuracy', 'x_axis_val', 'y_axis_val', 'z_axis_val'])
  dataframe = read_csv(filepath, skiprows = 1, header=None, names=['sensor_id','accuracy','x_axis_val', 'y_axis_val', 'z_axis_val'])
  dataframe_acc =  (dataframe[dataframe["sensor_id"] == 10]).values
  dataframe_acc = np.delete(dataframe_acc, [0,1], axis=1)
  dataframe_gyro =  (dataframe[dataframe["sensor_id"] == 4]).values
  dataframe_gyro = np.delete(dataframe_gyro, [0,1], axis=1)
  return dataframe_acc,dataframe_gyro

def get_max_size():
  mylist = [f for f in glob.glob("data/single/test/*.csv")]
  mylist += [f for f in glob.glob("data/single/train/*.csv")]
  mx_acc = 0
  mx_gyro = 0
  mn_acc = 800
  mn_gyro = 800
  for s in mylist:
    print(s)
    #cur_data = load_file(s)
    cur_data_acc, cur_data_gyro = load_file_seperate(s)
    ro_acc = cur_data_acc.shape[0]
    ro_gyro = cur_data_gyro.shape[0]
    #print(ro)
    mx_acc = max(mx_acc,ro_acc)
    mx_gyro = max(mx_gyro,ro_gyro)
    #mn = min(mn,ro)
  return mx_acc, mx_gyro

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

def load_file_padded(mx_acc_size, mx_gyro_size, filepath):
  data_acc, data_gyro = load_file_seperate(filepath)
  extra_acc = mx_acc_size - data_acc.shape[0]
  extra_gyro = mx_gyro_size - data_gyro.shape[0]
  for i in range(extra_acc):
    data_acc = np.append(data_acc,[[0.0,0.0,0.0]],axis=0)
  for i in range(extra_gyro):
    data_gyro = np.append(data_gyro,[[0.0,0.0,0.0]],axis=0)
  return data_acc, data_gyro


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
  loaded_acc = list()
  loaded_gyro = list()
  loaded_label = list()
  #mxa,mxg = get_max_size()
  mxa = 347
  mxg = 346
  #for generalizing change status above 3 lines
  scaler_acc = MinMaxScaler(feature_range=(-1,1))
  scaler_gyro = MinMaxScaler(feature_range=(-1,1))
  for name in filenames:
    data_acc, data_gyro = load_file_padded(mxa,mxg,prefix + name)
    scaler_acc.fit(data_acc)
    scaler_gyro.fit(data_gyro)
    data_acc_normalized = scaler_acc.transform(data_acc)
    data_gyro_normalized = scaler_gyro.transform(data_gyro)
    #print(name)
    label = get_label(name)
    loaded_acc.append(data_acc_normalized)
    loaded_gyro.append(data_gyro_normalized)
    loaded_label.append(label)
  X_acc = np.stack(loaded_acc,axis=0)
  X_gyro = np.stack(loaded_gyro,axis=0)
  y = np.stack(loaded_label,axis=0)
  y = np.reshape(y,(-1,1))
  return X_acc,X_gyro,y



def load_dataset():
  trainlist = [f for f in glob.glob("data/single/train/*.csv")]
  testlist = [f for f in glob.glob("data/single/test/*.csv")]

  trainX_acc, trainX_gyro, trainy = load_group(trainlist)
  testX_acc, testX_gyro, testy = load_group(testlist)

  trainy = trainy - 1
  testy = testy - 1
  #print(testy)
  trainy = to_categorical(trainy)
  testy = to_categorical(testy)
  #print(testy)
  #print(trainy.shape,testy.shape)
  return trainX_acc, trainX_gyro, trainy, testX_acc, testX_gyro, testy