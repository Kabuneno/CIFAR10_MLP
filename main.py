import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
import numpy as np
from activations import relu,soft_max
from tqdm import tqdm
from network import back_pass,forward_pass
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



cifar10_classess = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck"
]

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.reshape(x_train.shape[0],-1)
x_test_test = x_test.reshape(x_test.shape[0],-1)
x_train = x_train / 255.0
x_test_test = x_test_test / 255.0

np.random.seed(42)
W1 = np.random.randn(3072,512) * 0.01
b1 = np.zeros((1,512))
W2 = np.random.randn(512,256) * 0.01
b2 = np.zeros((1,256))
W3 = np.random.randn(256,10) * 0.01
b3 = np.zeros((1,10))

# Training
def update_params(dW1,db1,dW2,db2,dW3,db3,lr=0.1):
  global W1,b1,W2,b2,W3,b3
  W1 -= dW1 * lr
  b1 -= db1 * lr
  W2 -= dW2 * lr
  b2 -= db2 * lr
  W3 -= dW3 * lr
  b3 -= db3 * lr


def train(X,y,W1,b1,W2,b2,W3,b3, epochs= 20,batch_size = 32 , lr = 0.1):
  m = X.shape[0]
  for epoch in tqdm(range(epochs)):
    indices = np.random.permutation(m)
    X_shufle = X[indices]
    y_shufle = y[indices]
    for i in range(0,m,batch_size):
      X_batch = X_shufle[i:i+batch_size]
      y_batch = y_shufle[i:i+batch_size]

      Z1,A1,Z2,A2,Z3,A3 = forward_pass(X_batch,W1,b1,W2,b2,W3,b3)
      dW1,db1,dW2,db2,dW3,db3 = back_pass(X_batch,y_batch,Z1,A1,Z2,A2,Z3,A3,W3,W2)
      
      update_params(dW1,db1,dW2,db2,dW3,db3,)
    # if epoch % 5 ==0:
    #   print(cross_entrophy(forward_pass(X_batch,W1,b1,W2,b2,W3,b3)[-1],y_batch))

train(x_train,y_train,W1,b1,W2,b2,W3,b3)


# Testing
def accuracy_score(y_pred,y_true):
  res = 0
  array = [1 if y_pred[i] == y_true[i] else 0 for i in range(y_pred.shape[0])]
  for i in array:
    res+= i
  return 1 / y_pred.shape[0] * res


def get_whole_pred(X):

  Z1s = X @ W1 +b1
  A1s = relu(Z1s)


  Z2s = A1s @ W2 + b2
  A2s = relu(Z2s)

  Z3s = A2s @ W3 + b3
  A3s = soft_max(Z3s)

  res = A3s
  res_res = res.tolist()
  real_res = [ res_res[i].index(max(res_res[i]) ) for i in range(len(res_res))]
  real_res = np.array(real_res,dtype=np.uint8)
  return real_res

print("Accuracy is actually",accuracy_score(get_whole_pred(x_test_test),y_test)* 100,"%")



def get_pred(X):
  Z1s = X @ W1 +b1
  A1s = relu(Z1s)

  Z2s = A1s @ W2 + b2
  A2s = relu(Z2s)

  Z3s = A2s @ W3 + b3
  A3s = soft_max(Z3s)

  res = A3s

  res_res = res.tolist()
  res = res_res[0].index(max(res_res[0]))
  return res

def want_to_know(j,m):
  for i in range (j,m):
    plt.imshow(x_test[i])
    plt.axis('off')
    plt.show()

    if get_pred(x_test_test[i]) == y_test[i]:
      print(f"looks like {cifar10_classess[get_pred(x_test_test[i])] }")
      print("✅")
    else:
      print(f"looks like {cifar10_classess[get_pred(x_test_test[i])] } but its",cifar10_classess[y_test[i][0]])
      print("❌")



want_to_know(0,30)



y_true = y_test.flatten()
y_pred = get_whole_pred(x_test_test)
cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cifar10_classess)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.show()