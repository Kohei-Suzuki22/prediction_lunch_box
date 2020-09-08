import pandas as pd
import numpy as np
import seaborn as sns
import csv
import pdb
import ipdb
from scipy.stats import median_test
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression as LR
from sklearn.ensemble import RandomForestRegressor as RF

import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import SimpleRNN
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error as MSE
import time
import statistics
import scipy
from scipy import signal
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller



train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")
sample = pd.read_csv("./sample.csv",header=None)

train.index = pd.to_datetime(train["datetime"])


train.index = pd.to_datetime(train["datetime"])
train = train.reset_index(drop=True)
train["days"] = train.index
train["payday"] = train["payday"].fillna(0)
train["kcal"] = train["kcal"].fillna(-1)
train["precipitation"] = train["precipitation"].apply(lambda x : -1 if x == "--" else float(x)).astype(np.float)
train["event"] = train["event"].fillna("なし")
train["remarks"] = train["remarks"].fillna("なし")
train["month"] = train["datetime"].apply(lambda x : int(x.split("-")[1]))
train["amuse"] = train["remarks"].apply(lambda x : 1 if x == "お楽しみメニュー" else 0)
train["curry"] = train["name"].apply(lambda x : 1 if x.find("カレー") >= 0 else 0)
train["zeroRain"] = train["precipitation"].apply(lambda x : 1 if x == -1 else 0 )


input_var = ["y","kcal","days","precipitation","weather","amuse","curry","zeroRain","temperature","month"]
input_data = pd.get_dummies(train[input_var])


input_data.to_csv("analysis.csv")

input_y = train["y"].values
input_y = scipy.stats.zscore(input_y)


input_kcal = train["kcal"].values
input_kcal = scipy.stats.zscore(input_kcal)

input_days = train["days"].values
input_days = scipy.stats.zscore(input_days)

input_month = train["month"].values
input_month = scipy.stats.zscore(input_month)

input_curry = input_data["curry"].values
input_curry = scipy.stats.zscore(input_curry)

input_amuse = input_data["amuse"].values
input_amuse = scipy.stats.zscore(input_amuse)

input_zerorain = input_data["zeroRain"].values
input_zerorain = scipy.stats.zscore(input_zerorain)

input_temperature = input_data["temperature"].values
input_temperature = scipy.stats.zscore(input_temperature)

input_weather_fine = input_data["weather_快晴"].values
input_weather_fine = scipy.stats.zscore(input_weather_fine)

input_weather_sun = input_data["weather_晴れ"].values
input_weather_sun = scipy.stats.zscore(input_weather_sun)

input_weather_cloud = input_data["weather_曇"].values
input_weather_cloud = scipy.stats.zscore(input_weather_cloud)

input_weather_sin_cloud = input_data["weather_薄曇"].values
input_weather_sin_cloud = scipy.stats.zscore(input_weather_sin_cloud)

input_weather_rain = input_data["weather_雨"].values
input_weather_rain = scipy.stats.zscore(input_weather_rain)

input_weather_snow = input_data["weather_雪"].values
input_weather_snow = scipy.stats.zscore(input_weather_snow)

input_weather_thunder = input_data["weather_雷電"].values
input_weather_thunder = scipy.stats.zscore(input_weather_thunder)


days = input_data["days"].values


np.random.seed(777)

def rmse(y_true,y_pred):
    return round(np.sqrt(MSE(y_true,y_pred)),3)


affect_length = 16

def make_answers(y, affect_length):
  answers = np.array([])
  for i in range(len(y) - affect_length):
    answers = np.append(answers,y[i+affect_length])


  return answers


def make_input_data(y, affect_length):
  factors = []
  for i in range(len(y) - affect_length):
    factors.append(y[i:i+affect_length])

  factors = np.array(factors)
  return factors


n_in = 7
factors = np.empty((0,n_in))


for i in range(len(train)-1):
  factors = np.append(factors,np.array([[input_y[i],input_days[i+1],input_kcal[i+1],input_temperature[i+1],input_month[i+1],input_amuse[i+1],input_curry[i+1]]]),axis=0)

'''factors.shape => (206,7)'''






factors = make_input_data(factors,affect_length)
'''factors.shape => (190,16,7)'''
answers = make_answers(input_y[1:],(affect_length))


n_out = 1
n_hidden_unit = 100
lr = 0.0001


def make_model(factors,answers,input_dim):
  model = Sequential()
  model.add(SimpleRNN(n_hidden_unit, batch_input_shape=(None, affect_length, input_dim), return_sequences=False))
  # model.add(LSTM(n_hidden_unit, batch_input_shape=(None, affect_length, input_dim), return_sequences=False))

  model.add(Dense(output_dim=10,input_dim=input_dim))
  model.add(Activation("relu"))
  model.add(Dense(output_dim=1,input_dim=10))
  model.add(Activation("linear"))
  optimizer = Adam(lr = lr)
  model.compile(loss="mean_absolute_error", optimizer=optimizer)
  early_stopping = EarlyStopping(monitor='val_loss',  mode='auto', patience=30)
  ipdb.set_trace()
  model.fit(factors,answers,batch_size=1,epochs=1,validation_split=0.2, callbacks= [early_stopping])
  pred = model.predict(factors)
  return pred


def show_graph(x,pred,expect,affect_length):

  pred = pred.reshape(-1,1)
  x = x.reshape(-1,1)
  expect = expect.reshape(-1,1)

  print("---学習データ+検証データ---")
  print(rmse(expect[affect_length:],pred))

  print("---検証データ---")
  validation_size = int(round(len(expect) * 0.2,0))
  print(rmse(expect[affect_length:][-validation_size:],pred[-validation_size:]))


  '''通常グラフ'''
  plt.plot(x,expect, color='blue', label='expect')
  plt.plot(x[affect_length:],pred,color="red",label="pred")

  plt.xlabel('days')
  plt.ylabel('sold')
  plt.legend(loc='lower left')  # 図のラベルの位置を指定。
  plt.show()

  '''検証部分グラフ'''
  plt.xlim(150, 220)
  plt.plot(x,expect, color='blue', label='expect')
  plt.plot(x[affect_length:],pred,color="red",label="pred")

  plt.xlabel('days')
  plt.ylabel('sold')
  plt.legend(loc='lower left')  # 図のラベルの位置を指定。
  plt.show()

# answers = answers.reshape(-1,1)
# pred = make_model(factors,answers,n_in)
# show_graph(days[1:],pred,input_y[1:],affect_length)



def detrend(y,window):
  move_mean_y = y.rolling(window=window).mean()
  for i in range(window):
    if np.isnan(move_mean_y[i]):
      move_mean_y[i] = y[0:(i+1)].mean()

  ipdb.set_trace()
  detrend_y = y - move_mean_y
  detrend_y = scipy.stats.zscore(detrend_y)
  
  return detrend_y



detrend_y = detrend(train["y"],50)
detrend_y = detrend_y.reshape(-1,1)
detrend_answers = make_answers(detrend_y[1:],(affect_length))


'''分析用のデータを作成'''
train["detrend_y"] = detrend_y
ipdb.set_trace()

low_spike_train = train[train["detrend_y"] < -1]
# low_spike_train.to_csv("low_spike_train.csv")



pred = make_model(factors,detrend_answers,n_in)

# show_graph(days[1:],pred[1:],input_y[1:],affect_length)


#   plt.plot(x,y,label="original")
#   plt.plot(x,rolmean,label="moving ave")




'''トレンド除去グラフ'''

# show_graph(days[1:],pred,detrend_expect[1:],affect_length)



'''トレンド除去後、学習データと予測データの差分グラフ'''
# non_linear_y = np.insert(non_linear_y,0,np.zeros(affect_length))
# plt.plot(days,non_linear_y)
# plt.xlabel('days')
# plt.ylabel('sold')
# plt.legend(loc='lower left')  # 図のラベルの位置を指定。
# plt.show()





# print(input_y.shape)
# print(answers.shape)

# print(sample[1])


# batch_sizeは小さい方が正確。
# relu関数を使うとマイナスを考慮出来ないから、tanhを使っている。



'''
※ 学習中の loss, val_lossの違い

loss -> 学習用データを与えた際の損失値
        小さいほど学習出来たことを表す。

val_loss -> 検証用データを与えた際の損失値
            小さいほど正しい結果(検証データに近い値)を出せた。
'''


