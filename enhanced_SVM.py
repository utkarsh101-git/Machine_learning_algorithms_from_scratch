import numpy as np
class enhanced_svc:
  def __init__(self,  C=1.0, learning_rate=1.0 , batch_size=32):
    self.C = C
    self.lr = learning_rate
    self.batch_size = batch_size

  @staticmethod
  def hinge_loss( X, y, w, C):
    N = float(X.shape[0] )
    distances = 1- y.reshape((-1,1))*np.dot(X, w) 
    distances[distances < 0] = 0
    hinge = np.sum(distances )
    hinge = C*hinge
    total_loss = np.dot(w.T, w)/2 + hinge
    return total_loss

  @staticmethod
  def compute_gradients( X, y, w, C):
    N = X.shape[0]
    distance = 1 - y.reshape((-1,1)) * np.dot(X, w)
    #print(distance)
    grad = np.zeros(w.shape)
    for i, d in enumerate(distance):
      if(max(0,d) == 0):
        di = w
      else:
        di = w - C*y[i]*X[i,: ].reshape((-1,1))
        #print(y[i].shape)

      grad +=di 

    return grad 

  @staticmethod
  def get_mini_batches( X, y, batch_size):

    mini_batches = []
    ind = np.arange(0,X.shape[0])
    np.random.shuffle(ind)

    X = np.take(X, ind, axis=0)
    y = np.take(y, ind, axis=0)

    n_minibatches = X.shape[0] // batch_size
    
    
    c1 = 0
    c2 = n_minibatches

    for i in range(n_minibatches):
      x_mini = X[c1:c2,:]
      y_mini = y[c1:c2]
      
      mini_batches.append((x_mini, y_mini))
      c1 = c2
      c2 = c2 + n_minibatches

    return mini_batches

  
  def rbf_kernel_fit(self, x, gamma='auto'):
    n_features = x.shape[0]
    lm_features = self.landmark.shape[0]
    
    if(gamma == 'auto'):
      #self.g = 0.003
      self.g =  1/( n_features )
    else:
      self.g = gamma

    print(self.g)
    new = np.zeros((n_features, lm_features ))
    for i, lm in enumerate(self.landmark):
      new[:,i] = np.exp(  -self.g * np.sum( np.square(x - lm) , axis=1 ) )
    print(new.shape)
    return new

  def rbf_kernel_predict(self, x):
    n_features = x.shape[0]
    lm_features = self.landmark.shape[0]
    new = np.zeros((n_features, lm_features ))

    for i, lm in enumerate(self.landmark):
      new[:,i] = np.exp(  -self.g * np.sum( np.square(x - lm) , axis=1 ) )
    print(new.shape)
    return new



  def fit(self, X, y, epoch=10, kernel='no', gamma='auto'):
    if kernel =='yes': 
      self.landmark = X.copy()
      X = self.rbf_kernel_fit(X, gamma)
    
    self.w = np.random.random(size=(X.shape[1],1))

    min_w = np.array([])
    optimizer = AdamOptimizer(learning_rate=self.lr)
    MIN_VAL = 2147483647
    history  = dict()
    
    for ep in range(epoch):
      mini = enhanced_svc.get_mini_batches(X,y_train, self.batch_size)
  
      for x_mini, y_mini in mini:
        gradients = enhanced_svc.compute_gradients(x_mini, y_mini, self.w, self.C)
        self.w = optimizer.optimize(gradients, self.w)

      
      total_loss = float(enhanced_svc.hinge_loss(X,y, self.w, self.C))      
      
      if(MIN_VAL > total_loss ):
       
        min_w = self.w
        MIN_VAL = total_loss

      history[ ep ] = (total_loss , self.w )
      print(f"epoch {ep} loss is {float(enhanced_svc.hinge_loss(X,y, self.w, self.C))}")
    
    if(min_w.size !=0):
      self.w = min_w
    return history

  def predict(self, X, kernel='no'):
    if kernel == 'yes':
      X = self.rbf_kernel_predict(X)
    print(X.shape)
    res  = np.sign(np.dot(X,self.w))
    res[res==0] =1
    return res
    



class AdamOptimizer:
  def __init__(self, learning_rate=1.0, epsilon=10e-8, beta1=0.01, beta2=0.999):
    self.lr = learning_rate
    self.m = np.array([])
    self.s = np.array([])
    self.m_corr = np.array([])
    self.s_corr = np.array([])
    self.epsilon = epsilon
    self.lr = learning_rate
    self.beta1 = beta1
    self.beta2 = beta2 

  def optimize(self, gradients, w):
    if(self.m.size == 0):
      print("inside")
      self.m = np.zeros(w.shape)
      self.s = np.zeros(w.shape)

    self.m = self.beta1*self.m + (1 - self.beta1) * gradients
    self.s = self.beta2*self.s + (1 - self.beta2) * gradients * gradients 
    
    self.m_corr = self.m/(1 - self.beta1)
    self.s_corr = self.s/(1 - self.beta2)

    w = w - self.lr * self.m_corr/np.sqrt( self.s_corr + self.epsilon )

    return w  
