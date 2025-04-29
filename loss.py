import numpy as np
from activations import one_hot

def cross_entrophy(y_pred,y_true):
  m = y_pred.shape[0]
  res = - np.sum(one_hot(y_true) *  np.log(y_pred + 1e-9)) / m
  return res