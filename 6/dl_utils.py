import tensorflow as tf
import numpy as np

class Normalization():
    
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.build()
        
    def build(self):
        self.weights = []
        self.mu = tf.constant(np.array([[-0.35, 0.]]), dtype="float32") 
        self.sigma = tf.constant(np.array([[0.5, 0.04]]), dtype="float32")  

    def __call__(self, inputs):
        return (inputs-self.mu)/self.sigma

class Relu():
    
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.build()
        
    def add_weight(self, shape):
        weight_init = tf.random.normal(shape=shape, mean=0.0, stddev=0.05, dtype="float32") 
        return tf.Variable(initial_value=weight_init, trainable=True)
        
    def build(self):
        self.w = self.add_weight(shape=(self.input_dim, self.output_dim))
        self.b = self.add_weight(shape=(1, self.output_dim))
        self.weights = [self.w, self.b]

    def __call__(self, inputs):
        z = tf.matmul(inputs, self.w) + self.b
        return tf.math.maximum(0.,z) 
    
class Linear():
    
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.build()
        
    def add_weight(self, shape):
        weight_init = tf.random.normal(shape=shape, mean=0.0, stddev=0.05, dtype="float32")
        return tf.Variable(initial_value=weight_init, trainable=True)
        
    def build(self):
        self.w = self.add_weight(shape=(self.input_dim, self.output_dim))
        self.b = self.add_weight(shape=(1, self.output_dim))
        self.weights = [self.w, self.b]

    def __call__(self, inputs):
        z = tf.matmul(inputs, self.w) + self.b
        return z
    
class Sigmoid():
    
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.build()
        
    def add_weight(self, shape):
        weight_init = tf.random.normal(shape=shape, mean=0.0, stddev=0.05, dtype="float32") 
        return tf.Variable(initial_value=weight_init, trainable=True)
        
    def build(self):
        self.w = self.add_weight(shape=(self.input_dim, self.output_dim))
        self.b = self.add_weight(shape=(1, self.output_dim))
        self.weights = [self.w, self.b]

    def __call__(self, inputs):
        z = tf.matmul(inputs, self.w) + self.b
        return tf.math.sigmoid(z)
    
class Softmax():
    
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.build()
        
    def add_weight(self, shape):
        weight_init = tf.random.normal(shape=shape, mean=0.0, stddev=0.05, dtype="float32")
        return tf.Variable(initial_value=weight_init, trainable=True)
        
    def build(self):
        self.w = self.add_weight(shape=(self.input_dim, self.output_dim))
        self.b = self.add_weight(shape=(1, self.output_dim))
        self.weights = [self.w, self.b]

    def __call__(self, inputs):
        z = tf.matmul(inputs, self.w) + self.b
        u = tf.math.exp(z)
        return u/tf.math.reduce_sum(u, axis=1, keepdims=True)
    
#==================================================

class MeanSquaredError():
    
    def __init__(self, model, reg_param):
        self.model = model
        self.reg_param = reg_param

    def __call__(self, y_true, y_pred):
        regularization = 0
        for i in range(len(self.model.weights)):
            regularization = regularization + tf.reduce_sum(tf.math.square(self.model.weights[i]))
            
        return tf.math.reduce_mean(tf.math.square(y_true-y_pred)) + self.reg_param*regularization
    
class CategoricalCrossEntropy():
    
    def __call__(self, y_true, y_pred): 
  
        cross_loss = -tf.math.reduce_mean(tf.math.reduce_sum(y_true*tf.math.log(y_pred), axis=1), axis=0)
        
        return cross_loss 
    
class CategoricalCrossEntropyKL():
    
    def __init__(self, kl_div_param, dist_param, initial_policy):
        self.kl_div_param = kl_div_param
        self.dist_param = dist_param
        self.initial_policy = initial_policy

    def __call__(self, y_true, y_pred, x): 
    
        reinforce_loss = -tf.math.reduce_mean(tf.math.reduce_sum(y_true*tf.math.log(y_pred), axis=1), axis=0)
        
        ref_probs = self.initial_policy(x)
        ref_probs = tf.stop_gradient(ref_probs)

        kl_div = tf.math.reduce_mean(tf.math.reduce_sum(y_pred*(tf.math.log(y_pred)-tf.math.log(ref_probs)), axis=1), axis=0)
        #kl_div = tf.math.reduce_mean(tf.math.reduce_sum(ref_probs*(tf.math.log(ref_probs)-tf.math.log(y_pred)), axis=1), axis=0)
            
        return (1-self.dist_param)*reinforce_loss + self.dist_param*self.kl_div_param*kl_div 

#==================================================

class Adam():

    def __init__(self, model, learning_rate, beta_1, beta_2, epsilon):
        self.model = model
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon= epsilon
        self.stop_training = False
        self.build()
        
    def add_weight(self, shape):
        weight_init = tf.zeros(shape=shape, dtype="float32")
        return  tf.Variable(initial_value=weight_init, trainable=False)
    
    def build(self):
        self.weights = []
        for weight in self.model.weights:
            m = self.add_weight(shape=weight.shape)
            v = self.add_weight(shape=weight.shape)
            self.weights.append([m,v])
            
    def apply(self, grads, weights):
        for i in range(len(weights)):
            w = weights[i]
            grad_w = grads[i]
            m = self.weights[i][0]
            v = self.weights[i][1]
            self.weights[i][0].assign(self.beta_1*m + (1-self.beta_1)*grad_w)  
            self.weights[i][1].assign(self.beta_2*v + (1-self.beta_2)*grad_w*grad_w)
            m_ = (1/(1-self.beta_1))*self.weights[i][0]
            v_ = (1/(1-self.beta_2))*self.weights[i][1]
            weights[i].assign(w - self.learning_rate*m_/(tf.math.sqrt(v_)+self.epsilon))
        
#==================================================

class VFunction():
    
    def __init__(self):
        self.h1 = Normalization(input_dim=2, 
                                output_dim=2)
        self.h2 = Relu(input_dim=2, 
                       output_dim=128)
        self.h3 = Relu(input_dim=128, 
                        output_dim=128)
        self.h4 = Linear(input_dim=128, 
                         output_dim=1)
        self.build()
    
    def build(self):
        self.layers = [self.h1, self.h2, self.h3, self.h4]
        self.weights = []
        for layer in self.layers:
            for weight in layer.weights:
                self.weights.append(weight)
        
    def __call__(self, inputs):
        a1 = self.h1(inputs)
        a2 = self.h2(a1)
        a3 = self.h3(a2)
        y = self.h4(a3)
        return y
        
    def config(self, reg_param, epochs, learning_rate, beta_1, beta_2, epsilon):
        self.reg_param = reg_param
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.beta_1 = beta_1 
        self.beta_2 = beta_2 
        self.epsilon = epsilon
        self.loss = MeanSquaredError(model=self, reg_param=self.reg_param)
        self.optimizer = Adam(model=self, learning_rate=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.epsilon) 
        
    def train_step(self, X, Y):
        with tf.GradientTape() as tape:
            H = self(X)
            loss = self.loss(Y, H)
        grads = tape.gradient(loss, self.weights)
        self.optimizer.apply(grads, self.weights)
        
    def fit(self, X, Y, reg_param=1., epochs=1, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07):
        self.config(reg_param, epochs, learning_rate, beta_1, beta_2, epsilon)
        for epoch in range(self.epochs):
            self.train_step(tf.constant(X, dtype="float32"), tf.constant(Y, dtype="float32"))
            
    def predict(self, inputs):
        return self(tf.constant(inputs, dtype="float32")).numpy()
    
class BFunction():
    
    def __init__(self):
        self.h1 = Normalization(input_dim=2, 
                                output_dim=2)
        self.h2 = Sigmoid(input_dim=2, 
                          output_dim=64)
        self.h3 = Sigmoid(input_dim=64, 
                          output_dim=64)
        self.h4 = Linear(input_dim=64, 
                         output_dim=1)
        self.build()
    
    def build(self):
        self.layers = [self.h1, self.h2, self.h3, self.h4]
        self.weights = []
        for layer in self.layers:
            for weight in layer.weights:
                self.weights.append(weight)
        
    def __call__(self, inputs):
        a1 = self.h1(inputs)
        a2 = self.h2(a1)
        a3 = self.h3(a2)
        y = self.h4(a3)
        return y
        
    def config(self, reg_param, epochs, learning_rate, beta_1, beta_2, epsilon):
        self.reg_param = reg_param
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.beta_1 = beta_1 
        self.beta_2 = beta_2 
        self.epsilon = epsilon
        self.loss = MeanSquaredError(model=self, reg_param=self.reg_param)
        self.optimizer = Adam(model=self, learning_rate=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.epsilon) 
        
    def train_step(self, X, Y):
        with tf.GradientTape() as tape:
            H = self(X)
            loss = self.loss(Y, H)
        grads = tape.gradient(loss, self.weights)
        self.optimizer.apply(grads, self.weights)
        
    def fit(self, X, Y, reg_param=1., epochs=1, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07):
        self.config(reg_param, epochs, learning_rate, beta_1, beta_2, epsilon)
        for epoch in range(self.epochs):
            self.train_step(tf.constant(X, dtype="float32"), tf.constant(Y, dtype="float32"))
            
    def predict(self, inputs):
        return self(tf.constant(inputs, dtype="float32")).numpy()

class InitialPolicy():
    
    def __init__(self):
        self.h1 = Normalization(input_dim=2, 
                                output_dim=2)
        self.h2 = Sigmoid(input_dim=2, 
                          output_dim=32)
        self.h3 = Sigmoid(input_dim=32, 
                          output_dim=32)
        self.h4 = Softmax(input_dim=32, 
                          output_dim=3)
        self.build()
    
    def build(self):
        self.layers = [self.h1, self.h2, self.h3, self.h4]
        self.weights = []
        for layer in self.layers:
            for weight in layer.weights:
                self.weights.append(weight)
        
    def __call__(self, inputs):
        a1 = self.h1(inputs)
        a2 = self.h2(a1)
        a3 = self.h3(a2)
        y = self.h4(a3)
        return y
        
    def config(self, epochs, learning_rate, beta_1, beta_2, epsilon):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.beta_1 = beta_1 
        self.beta_2 = beta_2 
        self.epsilon = epsilon
        self.loss = CategoricalCrossEntropy()
        self.optimizer = Adam(model=self, learning_rate=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.epsilon) 
        
    def train_step(self, X, Y):
        with tf.GradientTape() as tape:
            H = self(X)
            loss = self.loss(Y, H)
        grads = tape.gradient(loss, self.weights)
        self.optimizer.apply(grads, self.weights)
        
    def fit(self, X, Y, epochs=1, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07):
        self.config(epochs, learning_rate, beta_1, beta_2, epsilon)
        for epoch in range(self.epochs):
            self.train_step(tf.constant(X, dtype="float32"), tf.constant(Y, dtype="float32"))
            
    def predict(self, inputs):
        return self(tf.constant(inputs, dtype="float32")).numpy()

class Policy():
    
    def __init__(self, initial_policy):
        self.initial_policy = initial_policy
        self.h1 = Normalization(input_dim=2, 
                                output_dim=2)
        self.h2 = Sigmoid(input_dim=2, 
                          output_dim=32)
        self.h3 = Sigmoid(input_dim=32, 
                          output_dim=32)
        self.h4 = Softmax(input_dim=32, 
                          output_dim=3)
        self.build()
    
    def build(self):
        self.layers = [self.h1, self.h2, self.h3, self.h4]
        self.weights = []
        for layer in self.layers:
            for weight in layer.weights:
                self.weights.append(weight)
        
    def __call__(self, inputs):
        a1 = self.h1(inputs)
        a2 = self.h2(a1)
        a3 = self.h3(a2)
        y = self.h4(a3)
        return y
            
    def config(self, kl_div_param, dist_param, epochs, learning_rate, beta_1, beta_2, epsilon):
        self.kl_div_param = kl_div_param
        self.dist_param = dist_param
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.beta_1 = beta_1 
        self.beta_2 = beta_2 
        self.epsilon = epsilon
        self.loss = CategoricalCrossEntropyKL(kl_div_param=self.kl_div_param, dist_param=self.dist_param, initial_policy=self.initial_policy)
        self.optimizer = Adam(model=self, learning_rate=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.epsilon) 
                
    def train_step(self, X, Y):
        with tf.GradientTape() as tape:
            H = self(X)
            loss = self.loss(Y, H, X)
        grads = tape.gradient(loss, self.weights)
        self.optimizer.apply(grads, self.weights)
        
    def fit(self, X, Y, kl_div_param=300., dist_param=1., epochs=1, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07):
        self.config(kl_div_param, dist_param, epochs, learning_rate, beta_1, beta_2, epsilon)
        for epoch in range(self.epochs):
            self.train_step(tf.constant(X, dtype="float32"), tf.constant(Y, dtype="float32"))
            
    def predict(self, inputs):
        return self(tf.constant(inputs, dtype="float32")).numpy()
    
    

