#Headers
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#Defining the class of Softmax
class Softmax(object):    

  def __init__(self):
    self.W = None
    self.b = None
    
  def get_loss_grads(self, X, y, reg, n_features, n_samples, n_classes):
    scores = np.dot(X, self.W)+self.b
    exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    probs = exp_scores/np.sum(exp_scores, axis=1, keepdims=True)
    correct_logprobs = -np.log(probs[np.arange(n_samples), y])
    loss = np.sum(correct_logprobs)/n_samples
    dscores = probs.copy()
    dscores[np.arange(n_samples),y] -= 1
    dscores /= n_samples
    dW = X.T.dot(dscores)  
    dW += reg*self.W
    db = np.sum(dscores, axis=0, keepdims=True)

    return loss, dW, db

  def train(self, X, y, learning_rate=1e-4, reg=0.5, num_iters=500):
       
    n_features, n_samples = X.shape[1], X.shape[0]   
    n_classes = 16
    if (self.W is None) & (self.b is None):
      np.random.seed(2016) # for reproducible results
      self.W = np.random.normal(loc=0.0, scale=1e-4, size=(n_features, n_classes))
      self.b = np.zeros((1, n_classes))
        
    for iter in range(num_iters):
      loss, dW, db = self.get_loss_grads(X, y, reg, n_features, n_samples, n_classes)
      self.W -= learning_rate*dW
      self.b -= learning_rate*db
        
        
  def train_early_stopping(self, X_train, y_train, X_val, y_val, learning_rate=1e-4, reg=0.5, 
                           early_stopping_rounds=200):
    n_features, n_samples = X_train.shape[1], X_train.shape[0]   
    n_classes = 16
    if (self.W is None) & (self.b is None):
      np.random.seed(2016)
      self.W = np.random.normal(loc=0.0, scale=1e-4, size=(n_features, n_classes))
      self.b = np.zeros((1, n_classes))
    best_val_accuracy = -1
    best_weights, best_bias = None, None
    no_improvement = 0
    keep_training = True
        
    while keep_training:
      loss, dW, db = self.get_loss_grads(X_train, y_train, reg, n_features, n_samples, n_classes)
      self.W -= learning_rate*dW
      self.b -= learning_rate*db
      val_accuracy = np.mean(self.predict(X_val)==y_val)
      print('val_accuracy ',val_accuracy)
      if val_accuracy>best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_weights, best_bias = self.W, self.b
        no_improvement = 0
      else:
        no_improvement += 1
        
      if no_improvement == early_stopping_rounds:
        self.W, self.b = best_weights, best_bias
        keep_training = False
      
      
  def predict(self, X):
    
    y_pred = np.exp(np.dot(X, self.W)+self.b)
    y_pred = np.argmax(y_pred, axis=1)
    
    return y_pred

#Taking Inputs
mat = scipy.io.loadmat('Indian_pines.mat')
mat_2 = scipy.io.loadmat('Indian_pines_corrected.mat')
mat_3 = scipy.io.loadmat('Indian_pines_gt.mat')

indian_pines_mat_1_data = mat["indian_pines"]
indian_pines_corrected_mat_2_data = mat_2["indian_pines_corrected"]
indian_pines_gt_3_data = mat_3["indian_pines_gt"]

x=indian_pines_corrected_mat_2_data
y=indian_pines_gt_3_data

#To 2D matrix
x=x.reshape((145*145,200))
temp=x.copy().astype(float)
mean=np.mean(x,axis=0)
dev=np.std(temp)
temp=(x-mean)/dev
x=temp
y=y.reshape((-1))
initshape=y.shape[0]

#Removing Zeroes
list_zero=[]
tempy=y
for d in range(x.shape[0]):
	if y[d]==0:
		list_zero.append(d)
X=np.delete(x,list_zero,0)
y=np.delete(y,list_zero,0)

for i in range(y.shape[0]):
	y[i]-=1

#Checking X and Y
print(X.shape)
print(y.shape)
X_train, y_train = X[0:8000], y[0:8000]
X_val, y_val = X[8000:], y[8000:]

#Declaring Classifier and actually training
softmax = Softmax()
softmax.train_early_stopping( X_train, y_train, X_val, y_val, learning_rate=0.5, reg=0.0, early_stopping_rounds=3000)

#Final accuracies after only SOFTMAX classification
print('Training accuracy',np.mean(softmax.predict(X_train)==y_train))
print('Validation accuracy',np.mean(softmax.predict(X_val)==y_val))

#Doing Discretization knowing the images are segmented.
ya=np.array([])
l=0
for o in range(21025):
	if o in list_zero:
		ya=np.append(ya,[0])
	else:
		ya=np.append(ya,softmax.predict(X[l]))
		l+=1
ya=ya.reshape(145,145)
plt.imshow(ya)
plt.show()
tempy=tempy.reshape(145,145)
plt.imshow(tempy)
plt.show()
for i in range(1,144):
	for j in range(1,144):
		r=np.array([])
		if(ya[i][j]!=0):
			r=np.append(r,ya[i-1][j-1])
			r=np.append(r,ya[i][j-1])
			r=np.append(r,ya[i-1][j])
			r=np.append(r,ya[i][j])
			r=np.append(r,ya[i+1][j])
			r=np.append(r,ya[i][j+1])
			r=np.append(r,ya[i+1][j+1])
			r=np.append(r,ya[i+1][j-1])
			r=np.append(r,ya[i-1][j+1])
			(values,counts) = np.unique(r,return_counts=True)
			ind=np.argmax(counts)
			if np.max(counts)>5:
				ya[i][j]=values[ind]
plt.imshow(ya)
plt.show()
