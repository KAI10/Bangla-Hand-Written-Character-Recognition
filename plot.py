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



v1_data = pd.read_csv('data/single/train/G5NZCJ017507206-config.json-ashik-right-v1-single-2020-03-27-12-07-58.csv', skiprows=1, 
                      header=None, names=['unix_timestamp', 'sensor_id', 'accuracy', 'x_axis_val', 'y_axis_val', 'z_axis_val'])

v1_acc_data = v1_data[v1_data["sensor_id"] == 10]
v1_gyro_data = v1_data[v1_data["sensor_id"] == 4]
# print(v1_acc_data.count(), v1_gyro_data.count())

v1_gyro_data.plot(x='unix_timestamp', y=['x_axis_val', 'y_axis_val', 'z_axis_val'])

v1_acc_data.plot(x='unix_timestamp', y=['x_axis_val', 'y_axis_val', 'z_axis_val'])

v11_data = pd.read_csv('data/single/train/G5NZCJ017507206-config.json-ashik-right-v11-single-2020-03-27-12-28-35.csv', skiprows=1, 
                      header=None, names=['unix_timestamp', 'sensor_id', 'accuracy', 'x_axis_val', 'y_axis_val', 'z_axis_val'])

v11_acc_data = v11_data[v1_data["sensor_id"] == 10]
v11_gyro_data = v11_data[v1_data["sensor_id"] == 4]
# print(v1_acc_data.count(), v1_gyro_data.count())

v11_acc_data.plot(x='unix_timestamp', y=['x_axis_val', 'y_axis_val', 'z_axis_val'])