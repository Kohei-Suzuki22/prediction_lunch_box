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


dat = pd.concat([train,test],sort=False).reset_index(drop=True)


dat.index = pd.to_datetime(dat["datetime"])
dat=dat.reset_index(drop=True)

dat["days"] = dat.index
dat["payday"] = dat["payday"].fillna(0)
dat["kcal"] = dat["kcal"].fillna(-1)
dat["precipitation"] = dat["precipitation"].apply(lambda x : -1 if x == "--" else float(x)).astype(np.float)
dat["event"] = dat["event"].fillna("なし")
dat["remarks"] = dat["remarks"].fillna("なし")
dat["month"] = dat["datetime"].apply(lambda x : int(x.split("-")[1]))
dat["amuse"] = dat["remarks"].apply(lambda x : 1 if x == "お楽しみメニュー" else 0)
dat["curry"] = dat["name"].apply(lambda x : 1 if x.find("カレー") >= 0 else 0)
dat["zeroRain"] = dat["precipitation"].apply(lambda x : 1 if x == -1 else 0 )
dat["y"] = dat["y"].fillna(0)





def change_day7(day):
    if day == "月":
        return "Mon"
    elif day == "火":
        return "Tues"
    elif day == "水":
        return "Wed"
    elif day == "木":
        return "Thur"
    elif day == "金":
        return "Fri"


def change_weather(weather):
    if weather == "快晴":
        return "Fine"
    elif weather == "晴れ":
        return "Sunny"
    elif weather == "曇":
        return "Cloudy"
    elif weather == "薄曇":
        return "ThinCloudy"
    elif weather == "雨":
        return "Rainy"
    elif weather == "雷電":
        return "Thunder"
    elif weather == "雪":
        return "Snowy"




def translation_remarks(remarks):
    if remarks == "なし":
        return "Nothing"
    elif remarks == "鶏のレモンペッパー焼（50食）、カレー（42食）":
        return "ChickenLemon_Curry"
    elif remarks == "酢豚（28食）、カレー（85食）":
        return "Subuta_Curry"
    elif remarks == "お楽しみメニュー":
        return "Amuse"
    elif remarks == "料理長のこだわりメニュー":
        return "Chef'sCommitment"
    elif remarks == "手作りの味":
        return "HomemadeTaste"
    elif remarks == "スペシャルメニュー（800円）":
        return "SpecialMenu"


def translation_event(event):
    if event == "なし":
        return "Nothing"
    elif event == "ママの会":
        return "Mom'sMeet"
    elif event == "キャリアアップ支援セミナー":
        return "CareerSupportSeminar"




dat["week"] = dat["week"].apply(lambda x : change_day7(x))
dat["weather"] = dat["weather"].apply(lambda x : change_weather(x))
dat["remarks"] = dat["remarks"].apply(lambda x : translation_remarks(x))
dat["event"] = dat["event"].apply(lambda x : translation_event(x))

dat = pd.get_dummies(dat)

# ipdb.set_trace()


elems_basic = ["y","soldout","kcal","payday","precipitation","temperature","days","month","amuse","curry","zeroRain"]
elems_week = ["y","week_Mon","week_Tues","week_Wed","week_Thur","week_Fri"]
elems_weather = ["y","weather_Fine","weather_Sunny","weather_Cloudy","weather_ThinCloudy","weather_Rainy","weather_Snowy","weather_Thunder"]
elems_remarks = ["y","remarks_Amuse","remarks_Nothing","remarks_SpecialMenu","remarks_HomemadeTaste",	"remarks_Chef'sCommitment","remarks_Subuta_Curry","remarks_ChickenLemon_Curry"]
elems_event = ["y","event_Nothing","event_CareerSupportSeminar","event_Mom'sMeet"]





dat.to_csv("analysis2.csv")

correlation_matrix_basic = dat[elems_basic].corr().round(2)
correlation_matrix_week = dat[elems_week].corr().round(2)
correlation_matrix_weather = dat[elems_weather].corr().round(2)
correlation_matrix_remarks = dat[elems_remarks].corr().round(2)
correlation_matrix_event = dat[elems_event].corr().round(2)



# sns.heatmap(data=correlation_matrix_basic, annot=True)
# plt.show()
# sns.heatmap(data=correlation_matrix_week, annot=True)
# plt.show()
# sns.heatmap(data=correlation_matrix_weather, annot=True)
# plt.show()
# sns.heatmap(data=correlation_matrix_remarks, annot=True)
# plt.show()
# sns.heatmap(data=correlation_matrix_event, annot=True)
# plt.show()


# ipdb.set_trace()
correlation_matrix_basic_150 = dat[150:][elems_basic].corr().round(2)
correlation_matrix_week_150 = dat[150:][elems_week].corr().round(2)
correlation_matrix_weather_150 = dat[150:][elems_weather].corr().round(2)
correlation_matrix_remarks_150 = dat[150:][elems_remarks].corr().round(2)
correlation_matrix_event_150 = dat[150:][elems_event].corr().round(2)


# sns.heatmap(data=correlation_matrix_basic_150, annot=True)
# plt.show()
# sns.heatmap(data=correlation_matrix_week_150, annot=True)
# plt.show()
# sns.heatmap(data=correlation_matrix_weather_150, annot=True)
# plt.show()
# sns.heatmap(data=correlation_matrix_remarks_150, annot=True)
# plt.show()
# sns.heatmap(data=correlation_matrix_event_150, annot=True)
# plt.show()


non_linear_elems = np.array([])

# ipdb.set_trace()

high_spike = dat[dat["days"].isin([155, 174, 181, 183, 196, 205])]
high_spike = high_spike[["y","soldout","kcal","payday","precipitation","temperature","days","month","amuse","curry","zeroRain","week_Mon","week_Tues","week_Wed","week_Thur","week_Fri","weather_Fine","weather_Sunny","weather_Cloudy","weather_ThinCloudy","weather_Rainy","weather_Snowy","weather_Thunder","remarks_Amuse","remarks_Nothing","remarks_SpecialMenu","remarks_HomemadeTaste",	"remarks_Chef'sCommitment","remarks_Subuta_Curry","remarks_ChickenLemon_Curry","event_Nothing","event_CareerSupportSeminar","event_Mom'sMeet"]]

high_spike = high_spike[high_spike["days"] > 150]
# ipdb.set_trace()

high_spike.to_csv("high_spike.csv")



low_spike = dat[dat["days"].isin([18,27,56,64,113,175,191])]
low_spike = low_spike[["y","soldout","kcal","payday","precipitation","temperature","days","month","amuse","curry","zeroRain","week_Mon","week_Tues","week_Wed","week_Thur","week_Fri","weather_Fine","weather_Sunny","weather_Cloudy","weather_ThinCloudy","weather_Rainy","weather_Snowy","weather_Thunder","remarks_Amuse","remarks_Nothing","remarks_SpecialMenu","remarks_HomemadeTaste",	"remarks_Chef'sCommitment","remarks_Subuta_Curry","remarks_ChickenLemon_Curry","event_Nothing","event_CareerSupportSeminar","event_Mom'sMeet"]]


# correlation_matrix_low_spike1 = low_spike.corr().round(2)


correlation_matrix_basic_high = low_spike[elems_basic].corr().round(2)
correlation_matrix_week_high = low_spike[elems_week].corr().round(2)
correlation_matrix_weather_high = low_spike[elems_weather].corr().round(2)
correlation_matrix_remarks_high = low_spike[elems_remarks].corr().round(2)
correlation_matrix_event_high = low_spike[elems_event].corr().round(2)

# sns.heatmap(data=correlation_matrix_basic_high, annot=True)
# plt.show()

# ipdb.set_trace()
# sns.heatmap(data=correlation_matrix_week_high, annot=True)
# plt.show()
# sns.heatmap(data=correlation_matrix_weather_high, annot=True)
# plt.show()
# sns.heatmap(data=correlation_matrix_remarks_high, annot=True)
# plt.show()
# sns.heatmap(data=correlation_matrix_remarks_high, annot=True)
# plt.show()




low_spike.to_csv("low_spike.csv")


# train = pd.read_csv("./train.csv")


# ipdb.set_trace()


# sns.boxplot(x="curry",y="y",data=high_spike)

# plt.show()



low_spike_train = pd.read_csv("./low_spike_train.csv")


# low_spike_train["week"] = low_spike_train["week"].apply(lambda x : change_day7(x))
# low_spike_train["weather"] = low_spike_train["weather"].apply(lambda x : change_weather(x))
# low_spike_train["remarks"] = low_spike_train["remarks"].apply(lambda x : translation_remarks(x))
# low_spike_train["event"] = low_spike_train["event"].apply(lambda x : translation_event(x))

# low_spike_train = pd.get_dummies(low_spike_train)

# ipdb.set_trace()
low_spike_train.to_csv("low_spike_train.csv")

low_spike_train = low_spike_train[["y","soldout","kcal","payday","precipitation","temperature","days","month","amuse","curry","zeroRain","week_Mon","week_Tues","week_Wed","week_Thur","week_Fri","weather_Fine","weather_Sunny","weather_Cloudy","weather_ThinCloudy","weather_Rainy","weather_Snowy","remarks_Nothing","remarks_ChickenLemon_Curry","event_Nothing","event_Mom'sMeet","detrend_y"]]


elems_basic = ["detrend_y","soldout","kcal","payday","precipitation","temperature","days","month","amuse","curry","zeroRain"]
elems_week = ["detrend_y","week_Mon","week_Tues","week_Wed","week_Thur","week_Fri"]
elems_weather = ["detrend_y","weather_Fine","weather_Sunny","weather_Cloudy","weather_ThinCloudy","weather_Rainy","weather_Snowy"]
elems_remarks = ["detrend_y","remarks_Nothing","remarks_ChickenLemon_Curry"]
elems_event = ["detrend_y","event_Nothing","event_Mom'sMeet"]



correlation_matrix_basic_high = low_spike_train[elems_basic].corr().round(2)
correlation_matrix_week_high = low_spike_train[elems_week].corr().round(2)
correlation_matrix_weather_high = low_spike_train[elems_weather].corr().round(2)
correlation_matrix_remarks_high = low_spike_train[elems_remarks].corr().round(2)
correlation_matrix_event_high = low_spike_train[elems_event].corr().round(2)

# sns.heatmap(data=correlation_matrix_basic_high, annot=True)
# plt.show()
# sns.heatmap(data=correlation_matrix_week_high, annot=True)
# plt.show()
# sns.heatmap(data=correlation_matrix_weather_high, annot=True)
# plt.show()
# sns.heatmap(data=correlation_matrix_remarks_high, annot=True)
# plt.show()
# sns.heatmap(data=correlation_matrix_remarks_high, annot=True)
# plt.show()



sns.boxplot(x="soldout",y="detrend_y",data=low_spike_train)
plt.show()


sns.boxplot(x="kcal",y="detrend_y",data=low_spike_train)
plt.show()

