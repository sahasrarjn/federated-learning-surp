import numpy as np
import pandas as pd
from sklearn import datasets

class Dataloader:
    def __init__(self, data_name):
        self.data_name = data_name
        if self.data_name == 'banana':
            self.data_path = 'data/banana_data.csv'
        else: self.data_path = None

    def read(self):
        '''Read csv file and return pandas dataframe'''
        if self.data_name == 'banana':
            df = pd.read_csv(self.data_path, header=None)
            df.loc(axis=1)[0] = df.loc(axis=1)[0].astype(int)
            df.loc(axis=1)[0] = df.loc(axis=1)[0].replace(-1, 0) # DEBUG : replaced -1 with 0
        elif self.data_name == 'diabetes':
            df = datasets.load_diabetes(as_frame=True)
            df = pd.concat([df.target, df.data], axis=1)
            df.columns = list(range(df.shape[1]))
        else:
            raise ValueError('No such data')
        return df

    def split(self, df, shuffle=True):
        '''Split dataframe into train and test with shuffle'''
        idx = np.arange(df.shape[0])
        if shuffle: 
            np.random.shuffle(idx)
        split = int(df.shape[0] * 0.8)
        train_df = df.iloc[idx[:split]]
        test_df = df.iloc[idx[split:]]
        return train_df, test_df
    
    def to_matrix(self, df):
        '''Convert dataframe to x, y'''
        x = df.loc(axis=1)[1:].values
        y = df.loc(axis=1)[0].values
        return x, y

    def get_data(self):
        df = self.read()
        train_df, test_df = self.split(df)
        x_train, y_train = self.to_matrix(train_df)
        x_test, y_test = self.to_matrix(test_df)
        print(f'Loaded {self.data_name} dataset!!')
        return x_train, y_train, x_test, y_test


        