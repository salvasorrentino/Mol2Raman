import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
import pickle


train = pd.read_pickle(r"data\raw\train.pickle")
test = pd.read_pickle(r"data\raw\test.pickle")

scaler = MinMaxScaler()

train_raman = np.array(train['RAMAN_SPECTRUM'].to_list())
test_raman = np.array(test['RAMAN_SPECTRUM'].to_list())

train_raman_scaled = scaler.fit_transform(train_raman)

test_raman_scaled = scaler.fit_transform(test_raman)

train['RAMAN_SPECTRUM_SCALED'] = list(train_raman_scaled)
test['RAMAN_SPECTRUM_SCALED'] = list(test_raman_scaled)

train.to_pickle(r"data\raw\train.pickle")
test.to_pickle(r"data\raw\test.pickle")

with open(r"models\scalers\min_max_scaler.pickle",'wb') as f:
    pickle.dump(scaler, f)
