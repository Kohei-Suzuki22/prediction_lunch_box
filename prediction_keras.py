import pandas as pd
import numpy as np
import seaborn as sns
import csv
# import pdb
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


train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")
sample = pd.read_csv("./sample.csv",header=None)

train.index = pd.to_datetime(train["datetime"])


train.index = pd.to_datetime(train["datetime"])
train = train.reset_index(drop=True)
train["days"] = train.index
train["payday"] = train["payday"].fillna(0)
train["precipitation"] = train["precipitation"].apply(lambda x : -1 if x == "--" else float(x)).astype(np.float)
train["event"] = train["event"].fillna("なし")
train["remarks"] = train["remarks"].fillna("なし")
train["month"] = train["datetime"].apply(lambda x : int(x.split("-")[1]))
train["amuse"] = train["remarks"].apply(lambda x : 1 if x == "お楽しみメニュー" else 0)
train["curry"] = train["name"].apply(lambda x : 1 if x.find("カレー") >= 0 else 0)
train["zeroRain"] = train["precipitation"].apply(lambda x : 1 if x == -1 else 0 )


input_var = ["y","days","precipitation","weather","amuse","curry","zeroRain","temperature"]
input_data = pd.get_dummies(train[input_var])


input_data.to_csv("analysis.csv")

input_y = train["y"].values
input_y = scipy.stats.zscore(input_y)
# print(input_y.shape) #(207,)

input_days = train["days"].values
input_days = scipy.stats.zscore(input_days)

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



input_data = input_data.values.astype(np.float)
input_data = scipy.stats.zscore(input_data)


# 温度と販売数に相関。　温度高い → 販売数低い。
# 給料日は販売数が上がる。
# 日にちが経過するごとに販売数が低下。
# お楽しみメニューがある日は売上が上がる。
# お楽しみメニューの中でも、特にカレーが含まれるメニューの日は売り上げが良い。


np.random.seed(777)

# 二乗平均平方根誤差
def rmse(y_true,y_pred):
    return round(np.sqrt(MSE(y_true,y_pred)),3)



affect_length = 16


def make_answers(y, affect_length):
  answers = np.array([])
  for i in range(len(y) - affect_length):
    answers = np.append(answers,y[i+affect_length])

  return answers


def make_input_data(y, affect_length):
  factors = np.array([[]])
  for i in range(len(y) - affect_length):
    factors = np.append(factors,y[i:i+affect_length])

  factors = factors.reshape(-1,affect_length,1)
  return factors


answers = make_answers(input_y,affect_length)
factors_y = make_input_data(input_y,affect_length)
d = make_input_data(input_days,affect_length)
curry = make_input_data(input_curry,affect_length)
amuse = make_input_data(input_amuse,affect_length)
zeroRain = make_input_data(input_zerorain,affect_length)
temperature = make_input_data(input_temperature,affect_length)

weather_fine = make_input_data(input_weather_fine,affect_length)
weather_sun = make_input_data(input_weather_sun,affect_length)
weather_cloud = make_input_data(input_weather_cloud,affect_length)
weather_sin_cloud = make_input_data(input_weather_sin_cloud,affect_length)
weather_rain = make_input_data(input_weather_rain,affect_length)
weather_snow = make_input_data(input_weather_snow,affect_length)
weather_thunder = make_input_data(input_weather_thunder,affect_length)


def reshape_input(data):
  return data.reshape(-1)


factors_y = reshape_input(factors_y)
d = reshape_input(d)
curry = reshape_input(curry)
amuse = reshape_input(amuse)
zeroRain = reshape_input(zeroRain)
temperature = reshape_input(temperature)

weather_fine = reshape_input(weather_fine)
weather_sun = reshape_input(weather_sun)
weather_cloud = reshape_input(weather_cloud)
weather_sin_cloud = reshape_input(weather_sin_cloud)
weather_rain = reshape_input(weather_rain)
weather_snow = reshape_input(weather_snow)
weather_thunder = reshape_input(weather_thunder)


factors = np.array([[]])

# for i in range(len(factors_y)):
#   factors = np.append(factors,[factors_y[i],curry[i],amuse[i],zeroRain[i],temperature[i],weather_fine[i],weather_sun[i],weather_cloud[i],weather_sin_cloud[i],weather_rain[i],weather_snow[i],weather_thunder[i]])
for i in range(len(factors_y)):
  factors = np.append(factors,[factors_y[i],curry[i],amuse[i],zeroRain[i],temperature[i],weather_fine[i],weather_sun[i],weather_cloud[i],weather_sin_cloud[i],weather_rain[i],weather_snow[i],weather_thunder[i]])



# for i in range(len(factors_y)):
#   factors = np.append(factors,[factors_y[i],curry[i],amuse[i],zeroRain[i],temperature[i],weather_fine[i],weather_sun[i],weather_cloud[i],weather_rain[i]])

# for i in range(len(factors_y)):
#   factors = np.append(factors,[factors_y[i],curry[i],amuse[i],zeroRain[i],temperature[i]])



# for i in range(len(factors_y) -1):
#   factors = np.append(factors,[factors_y[i+1],curry[i],amuse[i],zeroRain[i],temperature[i],weather_fine[i],weather_sun[i],weather_cloud[i],weather_sin_cloud[i],weather_rain[i],weather_snow[i],weather_thunder[i]])

n_in = 12

factors = factors.reshape(-1,affect_length,n_in)
# print(factors.shape)


# factors_y = factors_y.reshape(-1,affect_length,1)





n_out = 1
n_hidden_unit = 100
lr = 0.0001


def make_model(factors,answers,input_dim):
  model = Sequential()
  model.add(SimpleRNN(n_hidden_unit, batch_input_shape=(None, affect_length, input_dim), return_sequences=False))

  model.add(Dense(output_dim=10,input_dim=input_dim))
  model.add(Activation("tanh"))
  model.add(Dense(output_dim=1,input_dim=10))
  model.add(Activation("linear"))
  optimizer = Adam(lr = lr)
  model.compile(loss="mean_absolute_error", optimizer=optimizer)
  early_stopping = EarlyStopping(monitor='val_loss',  mode='auto', patience=30)

  model.fit(factors,answers,batch_size=1,epochs=1,validation_split=0.2, callbacks= [early_stopping])
  # model.fit(factors,answers,batch_size=1,epochs=4000,validation_split=0.2, callbacks= [early_stopping])

  return model

answers = signal.detrend(answers)
answers = answers.reshape(-1,1)

model = make_model(factors,answers,n_in)
pred = model.predict(factors)

# ipdb.set_trace()



def show_graph(x,pred,expect,affect_length):

  pred = pred.reshape(-1,1)
  x = x.reshape(-1,1)
  detrend_expect = signal.detrend(expect)
  expect = expect.reshape(-1,1)
  detrend_expect = detrend_expect.reshape(-1,1)



  print("---学習データ+検証データ---")
  # print(rmse(expect,pred))
  print(rmse(detrend_expect[affect_length:],pred))

  print("---検証データ---")
  validation_size = int(round(len(expect) * 0.2,0))
  print(rmse(detrend_expect[affect_length:][-validation_size:],pred[-validation_size:]))
  # plt.xlim(150, 220)
  plt.plot(x,detrend_expect, color='blue', label='expect')
  plt.plot(x[affect_length:], pred, color='red', label='pred')
  plt.xlabel('days')
  plt.ylabel('sold')
  plt.legend(loc='lower left')  # 図のラベルの位置を指定。
  # ipdb.set_trace()
  plt.show()




# show_graph(days,pred,input_y,affect_length)
# show_graph(days,pred,answers,affect_length)

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

## トレンド除去

def show_y_trend(x,y):
  y = y.reshape(-1)
  x = x.reshape(-1)

  yd = signal.detrend(y)

  plt.xlabel('days')
  plt.ylabel('sold')
  plt.legend(loc='lower left')  # 図のラベルの位置を指定。

  plt.plot(x,y, color='blue', label='before Detrending')
  plt.plot(x,yd, color='red', label='after Detrending')
  plt.show()


# show_y_trend(days,input_y)








# ## テスト




# train["t"] = 1
# test["t"] = 0

dat = pd.concat([train,test],sort=False).reset_index(drop=True)


dat.index = pd.to_datetime(dat["datetime"])
dat=dat.reset_index(drop=True)




dat["days"] = dat.index
dat["payday"] = dat["payday"].fillna(0)
dat["precipitation"] = dat["precipitation"].apply(lambda x : -1 if x == "--" else float(x)).astype(np.float)
dat["event"] = dat["event"].fillna("なし")
dat["remarks"] = dat["remarks"].fillna("なし")
dat["month"] = dat["datetime"].apply(lambda x : int(x.split("-")[1]))
dat["amuse"] = dat["remarks"].apply(lambda x : 1 if x == "お楽しみメニュー" else 0)
dat["curry"] = dat["name"].apply(lambda x : 1 if x.find("カレー") >= 0 else 0)
dat["zeroRain"] = dat["precipitation"].apply(lambda x : 1 if x == -1 else 0 )
dat["y"] = dat["y"].fillna(0)

# input_var = ["days","precipitation","weather","amuse","curry","zeroRain","temperature"]
input_var = ["y","precipitation","weather","amuse","curry","zeroRain","temperature"]

input_data = pd.get_dummies(dat[input_var])


# input_data.to_csv("analysis.csv")
# ipdb.set_trace()
input_y = input_data["y"].values
# input_y = scipy.stats.zscore(input_y)
# # print(input_y.shape) #(207,)

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


# days = input_data["days"].values


# # print(dat)
answers = make_answers(input_y,affect_length)
factors_y = make_input_data(input_y,affect_length)
curry = make_input_data(input_curry,affect_length)
amuse = make_input_data(input_amuse,affect_length)
zeroRain = make_input_data(input_zerorain,affect_length)
temperature = make_input_data(input_temperature,affect_length)

weather_fine = make_input_data(input_weather_fine,affect_length)
weather_sun = make_input_data(input_weather_sun,affect_length)
weather_cloud = make_input_data(input_weather_cloud,affect_length)
weather_sin_cloud = make_input_data(input_weather_sin_cloud,affect_length)
weather_rain = make_input_data(input_weather_rain,affect_length)
weather_snow = make_input_data(input_weather_snow,affect_length)
weather_thunder = make_input_data(input_weather_thunder,affect_length)



# # new



factors = np.array([[]])


for i in range(len(factors_y)):
  factors = np.append(factors,[factors_y[i],curry[i],amuse[i],zeroRain[i],temperature[i],weather_fine[i],weather_sun[i],weather_cloud[i],weather_sin_cloud[i],weather_rain[i],weather_snow[i],weather_thunder[i]])



# for i in range(len(factors_y) -1):
#   factors = np.append(factors,[factors_y[i+1],curry[i],amuse[i],zeroRain[i],temperature[i],weather_fine[i],weather_sun[i],weather_cloud[i],weather_sin_cloud[i],weather_rain[i],weather_snow[i],weather_thunder[i]])

# print(factors[-1].shape)

# print(pred[-1])


factors = factors.reshape(-1,affect_length,n_in)





start_y = pred[-affect_length:].reshape(1,affect_length)[0]

# ipdb.set_trace()

test_y = np.array([])
ipdb.set_trace()

for i in range(40):
  # start_dat = len(dat) - len(test) - affect_length + (i+1)
  start_dat = len(dat) - len(test) - affect_length + (i)
  predicted = model.predict(factors[start_dat:(start_dat+1)])
  target = factors[start_dat+affect_length][0][0]
  if target == 0:
    factors[start_dat+affect_length][0][0] = predicted[0][0]
    test_y = np.append(test_y,predicted[0][0])
    print(start_dat)
  else:
    print("no")

print("hello.")

ipdb.set_trace()


  # test["y"][i] = predicted
  # start = np.append


# model = make_model(factors,answers,12)



# pred = model.predict(factors)
# print(answers.shape)


