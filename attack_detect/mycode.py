import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 

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

class MyPlot:
    def __init__(self):
        self.figsize = (8,8)

    def plot_loss(self, loss):
        plt.figure(self.figsize)
        plt.xlabel("iteration")
        plt.ylabel("loss")
        plt.plot(loss, color = 'r')


