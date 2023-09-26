import pandas as pd
import numpy as np
import datetime
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data_dir, sequence_length):
        df = pd.read_csv(data_dir, delimiter=";")
        df["clock"] = df["clock"].apply(lambda x: datetime.datetime.fromtimestamp(x).strftime('%Y/%m/%d %H:%M:%S'))
        df = df.sort_values("clock")

        self.X, self.y = create_sequences(df["value_avg"], sequence_length)
        MIN = self.X.min()
        MAX = self.X.max()
        self.X = MinMaxScale(self.X, MIN, MAX)
        self.y = MinMaxScale(self.y, MIN, MAX)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data)-seq_length):
        x = data.iloc[i:(i+seq_length)]
        y = data.iloc[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def MinMaxScale(array, min, max):
    return (array - min) / (max - min)