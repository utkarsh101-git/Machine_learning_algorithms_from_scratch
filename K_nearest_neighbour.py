import sys
import numpy as np
class ShapeError(Exception):
  pass

class knn:
  def __init__(self,neighbour=5,distance_type='elucidean'):
    self.k=neighbour
    self.dist_type=distance_type

  def single_prediction(self,x): #private
    y_distances = self.distance(x, self.X_train)
    neighbours=[]
    
    for i in range(self.k ):
      ind = y_distances.argmin()
      neighbours.append( self.y_train[ind] )  #y_train must be in single dim
      y_distances[ind] = sys.maxsize 

    return np.bincount(neighbours).argmax()
      
  def distance(self,x, y): #protected 
      if( self.dist_type is 'elucidean'):
        return np.sqrt( np.sum( np.square(x - y), axis=1) )
      
      elif (self.dist_type is 'manhattan'):
        return np.sum(np.abs(x - y), axis=1)
          

  def fit(self,X_train, y_train):
    self.X_train = X_train
    try:
      if(len(y_train.shape) is not 1):
        raise ShapeError("ShapeError: required shape is (n,) got {}".format(y_train.shape))
    
    except ShapeError as e:
      print(e)
      print("model training haulted")

    else:    
      print("model fit successfull")  
      self.y_train = y_train
    
    

  def predict(self,X_test):
    X_test = np.array(X_test)
    y_pred = np.array( [ self.single_prediction(x) for x in X_test ] )
    return y_pred