import pandas as pd
class MyData:
    def __init__(self):
        self.train = None
        self.test = None
        pass

    def get_data(self):
        self.train = pd.read_csv("./data/KDDTrain+.csv", header = None)
        self.test = pd.read_csv("./data/KDDTest+.csv", header = None)

    def one_hot_encode(self):
        encoded_train = pd.get_dummies(self.train[[1,2,3]], columns = [1,2,3])
        encoded_test = pd.get_dummies(self.test[[1,2,3]],columns = [1,2,3])
        missing_cols = set(encoded_train.columns) - set(encoded_test.columns)
        for col in missing_cols:
            encoded_test[col] = 0
        
        encoded_test = encoded_test[encoded_train.columns]
        encoded_train = encoded_train.astype(int)
        encoded_test = encoded_test.astype(int)
        return encoded_train, encoded_test

    def label_to_int(self):
        train_label = self.train.iloc[:, -2]
        test_label = self.test.iloc[: ,-2]
        y_train = train_label.map(lambda x: 0 if x == 'normal' else 1)
        y_test = test_label.map(lambda x: 0 if x == 'normal' else 1)
        return y_train, y_test

