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

import dataloader
import plot
import model

mymodel = run_training()

# trainX_acc, trainX_gyro, trainy, testX_acc, testX_gyro, testy = load_dataset() #ready to go into rnn
# ynew = model.predict([testX_acc,testX_gyro])
# print(ynew[0])
# print(testy[0])

# print(ynew[7])
# print(testy[7])