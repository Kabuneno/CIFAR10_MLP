import numpy as np

def relu(Z):
  return np.maximum(0,Z)

def relu_derivative(Z):
  return Z>0

def soft_max(Z):
  Z_stable = Z - np.max(Z,axis=1,keepdims= True)
  res = np.exp(Z_stable) / np.sum(np.exp(Z_stable),axis=1,keepdims=True)
  return res

def one_hot(y):
  res = np.zeros((y.shape[0],10 ))
  for i in range (y.shape[0]):
    res[i,y[i]] = 1
  return res
