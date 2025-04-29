import numpy as np
from activations import relu,soft_max, one_hot,relu_derivative
from tqdm import tqdm
from loss import cross_entrophy


def forward_pass(X,W1,b1,W2,b2,W3,b3):
  Z1 = X @ W1 +b1
  A1 = relu(Z1)

  Z2 = A1 @ W2 + b2
  A2 = relu(Z2)

  Z3 = A2 @ W3 + b3
  A3 = soft_max(Z3)
  return Z1, A1 , Z2 ,A2 , Z3,A3

def back_pass(X,y, Z1, A1 , Z2 ,A2,Z3,A3, W3,W2):
  m = X.shape[0]
  DZ3 = A3 - one_hot(y)
  DW3 = A2.T @ DZ3 / m
  DB3 = np.sum(DZ3,axis=0,keepdims=True)/m

  DA2 = DZ3 @ W3.T
  DZ2 = DA2 * relu_derivative(Z2)
  DW2 = A1.T @  DZ2 / m
  DB2 = np.sum(DZ2,axis=0,keepdims=True)/m

  DA1 = DZ2 @ W2.T
  DZ1 = DA1 * relu_derivative(Z1)
  DW1 = X.T @ DZ1/m
  DB1 = np.sum(DZ1,axis=0,keepdims=True)/m

  return DW1,DB1,DW2,DB2,DW3,DB3




