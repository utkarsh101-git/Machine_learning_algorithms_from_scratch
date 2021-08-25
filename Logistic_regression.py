

import numpy as np

class Logistic_Regression:
    
    #Constructor of the class
    def __init__(self,itr=100):
      self.n_epoch = itr
      self.t0 = itr/10
      self.t1 = itr

    def fit(self,X,y):
      m,n = X.shape
      #intitializing the weights to zero
      self.theta = np.zeros((n, 1))

      #using stochastic gradient descent algorithm
      for epoch in range(self.n_epoch):
        for i in range(m):

          #random indexes from the data
          rand_ind = np.random.randint(m)
          xi = X[rand_ind : rand_ind+1]
          yi = y[rand_ind : rand_ind+1]
          
          #gradients of the loss function
          grad = xi.T.dot(Logistic_Regression.sigmoid(xi.dot(self.theta) - yi) )
          #step size
          step_size = grad*self.lr_schedule(epoch*m + i)
          #updating the values of weights
          self.theta -= step_size
      print("model fitted")
    
    def predict(self,X):
      X = np.array(X)
      y_pred = Logistic_Regression.sigmoid(X.dot(self.theta) )
      
      for i in range(len(y_pred)):
        if(y_pred[i] < 0.5):
          y_pred[i] = 0
        else:
          y_pred[i] = 1
      return y_pred
    
    #learing_rate rescheduling
    def lr_schedule(self, t):
      return self.t0 / (t+self.t1)
    
    #finalized parameters of the model
    def params(self):
        return {'weights': self.theta, "epochs": self.n_epoch}                
    
    @staticmethod
    def sigmoid(X):
      return 1 / (1 + np.exp(-X))

    

    