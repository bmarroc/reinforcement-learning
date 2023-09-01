#!/usr/bin/env python

"""Class that defines the neural network model for the policy.
"""

import tensorflow as tf
import numpy as np

class Layer1():
    
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.build()
        
    def build(self):
        self.weights = []
        self.mu = tf.constant(np.array([[-1.2, -0.07]]), dtype="float32") 
        self.sigma = tf.constant(np.array([[1.7, 0.14]]), dtype="float32") 

    def __call__(self, inputs):
        return (inputs-self.mu)/self.sigma

class Layer2():
    
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.build()
        
    def add_weight(self, shape):
        weight_init = tf.random.normal(shape=shape, mean=0.0, stddev=0.05, dtype="float32") 
        return tf.Variable(initial_value=weight_init, trainable=True)
        
    def build(self):
        self.w = self.add_weight(shape=(self.output_dim, self.input_dim))
        self.b = self.add_weight(shape=(self.output_dim, 1))
        self.weights = [self.w, self.b]

    def __call__(self, inputs):
        z = tf.matmul(self.w, tf.transpose(inputs)) + self.b
        u = tf.transpose(z)
        return tf.math.sigmoid(u) 
    
class Layer3():
    
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.build()
        
    def add_weight(self, shape):
        weight_init = tf.random.normal(shape=shape, mean=0.0, stddev=0.05, dtype="float32")
        return tf.Variable(initial_value=weight_init, trainable=True)
        
    def build(self):
        self.w = self.add_weight(shape=(self.output_dim, self.input_dim))
        self.b = self.add_weight(shape=(self.output_dim, 1))
        self.weights = [self.w, self.b]

    def __call__(self, inputs):
        z = tf.matmul(self.w, tf.transpose(inputs)) + self.b
        v = tf.transpose(z)
        u = tf.math.exp(v)
        return u/tf.math.reduce_sum(u, axis=1, keepdims=True)

# -------------------------------------------------- #
    
class LossFunction():
    
    def __init__(self, model):
        self.model = model

    def __call__(self, y_true, y_pred):
        regularization = 0
        for i in range(len(self.model.weights)):
            regularization = regularization + tf.reduce_sum(tf.math.square(self.model.weights[i]))
            
        return -tf.math.reduce_mean(tf.math.reduce_mean(y_true*tf.math.log(y_pred), axis=1), axis=0) + regularization

# -------------------------------------------------- #    
    
class Optimizer():

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
            
    def train_step(self, X, Y):
        with tf.GradientTape() as tape:
            H = self.model(X)
            loss = self.model.loss(tf.stop_gradient(Y), H)
        grads = tape.gradient(loss, self.model.weights)
        self.apply(grads, self.model.weights)
        
# -------------------------------------------------- # 
        
class Policy():
    
    def __init__(self):
        self.build()
    
    def build(self):
        self.h1 = Layer1(input_dim=2, 
                         output_dim=2)
        self.h2 = Layer2(input_dim=2, 
                         output_dim=32)
        self.h2_ = Layer2(input_dim=32, 
                          output_dim=32)
        self.h3 = Layer3(input_dim=32, 
                         output_dim=3)
        self.layers = [self.h1, self.h2, self.h2_, self.h3]
        self.weights = []
        for layer in self.layers:
            for weight in layer.weights:
                self.weights.append(weight)
        
    def __call__(self, inputs):
        a1 = self.h1(inputs)
        a2 = self.h2(a1)
        a2_ = self.h2_(a2)
        y = self.h3(a2_)
        return y
        
    def train_setup(self, epochs, learning_rate, beta_1, beta_2, epsilon, verbose):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.beta_1 = beta_1 
        self.beta_2 = beta_2 
        self.epsilon = epsilon
        self.verbose = verbose
        self.loss = LossFunction(model=self)
        self.optimizer = Optimizer(model=self, learning_rate=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.epsilon) 
        
    def fit(self, X, Y, epochs=1, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, verbose=False):
        self.train_setup(epochs, learning_rate, beta_1, beta_2, epsilon, verbose)
        if verbose:
            print('Train on {} samples'.format(X.shape[0]))
        for epoch in range(self.epochs):
            self.optimizer.train_step(tf.constant(X, dtype="float32"), tf.constant(Y, dtype="float32"))
            
    def predict(self, inputs):
        return self(tf.constant(inputs, dtype="float32")).numpy()