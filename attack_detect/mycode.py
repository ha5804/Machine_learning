import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np 
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
    
    def binary_label(self):
        y = self.label_to_int()
        y_binary = (y != 11).astype(int)
        return y_binary
    
data = MyData()
data.get_data()
x = data.encode()
y = data.binary_label()
# print(x)
# print(y.value_counts())
# print(y)
print(y)

class Model:
    def __init__(self):
        self.weight = np.zeros((84, 1))
        pass

    def predict(self, x):
        feature_matrix = x
        z = feature_matrix @ self.weight
        pred = self.activate(z)
        return pred
    
    def activate(self, x):
        y = 1/ (1+np.exp(-x))
        return y
    
    def update(self, weight):
        self.weight = weight
    
class MyOptim:
    def __init__(self, model):
        self.model  = model
        self.eps    = 1e-8
        self.lr = 0.1
    def compute_loss(self, label, prediction):
        # ===========================
        eps = self.eps
        loss = (1 / (len(label))) * ((((-1) * label.T) @ np.log(prediction + eps)) - ((np.ones_like(label) - label).T @ np.log(np.ones_like(label) - prediction + eps)))
        # ===========================
        return loss
     
    def compute_residual(self, label, prediction):
        # ===========================
        res = label - prediction
        # ===========================
        return res
   
    def compute_gradient(self, x, label, pred):
        # ===========================
        A = x
        grad = (1 / (len(label))) * (A.T @ (pred - label))
        # ===========================
        return grad
        
    def step(self, grad):
        # ===========================
        weight = self.model.weight
        next_weight = weight - (self.lr * grad)
        self.model.update(next_weight)
        pass 
        # ===========================



def train(x, y):
    model = Model()
    optim = MyOptim(model)
    loss_lis = []
    y = y.values.reshape(-1,1)
    for _ in range(100):
        predict = model.predict(x)
        loss = optim.compute_loss(y, predict)
        grad = optim.compute_gradient(x, y, predict)
        optim.step(grad)
        if _ % 10 == 0:
            loss_lis.append(loss)
    return loss_lis

ls = train(x, y)
for i in ls:
    print(i)
