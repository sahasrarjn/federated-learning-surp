import numpy as np
import pandas as pd

class Dataloader:
    def __init__(self, data_path):
        self.data_path = data_path

    def read(self):
        '''Read csv file and return pandas dataframe'''
        df = pd.read_csv(self.data_path, header=None)
        df.loc(axis=1)[0] = df.loc(axis=1)[0].astype(int)
        df.loc(axis=1)[0] = df.loc(axis=1)[0].replace(-1, 0) # DEBUG : replace -1 with 0
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
        return x_train, y_train, x_test, y_test



        