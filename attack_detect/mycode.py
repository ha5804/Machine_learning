import pandas as pd

class MyData:
    def __init__(self):
        self.train = None
        self.test = None

    def get_data(self):
        self.train = pd.read_csv("./attack_detect/data/KDDTrain+.csv", header = None)
        self.test = pd.read_csv("./attack_detect/data/KDDTest+.csv", header = None)
    
    def encode(self, feature_number = [1,2,3]):
        selected = self.train[feature_number]
        encoded = pd.get_dummies(selected, feature_number)
        encoded = encoded.astype(int) #저장해야 정수로변환
        return encoded #x
    
    def get_label(self):
        label = self.train.iloc[:, -2]
        return label 

    def label_to_int(self):
        label = self.get_label()
        unique_labels = sorted(label.unique())
        label_int = {label: idx for idx, label in enumerate(unique_labels)}
        y = label.map(label_int)
        return y 
    
data = MyData()
data.get_data()
x = data.encode()
y = data.label_to_int()
print(x)
print(y.value_counts())
print(y)





