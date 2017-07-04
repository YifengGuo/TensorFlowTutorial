# import tensorflow as tf 
# sess = tf.Session()

# # TensorFlow provides optimizers that slowly change each variable in order to minimize the loss function. 
# # The simplest optimizer is gradient descent. It modifies each variable according to the magnitude of the derivative of loss with respect to that variable. 
# # In general, computing symbolic derivatives manually is tedious and error-prone. Consequently, TensorFlow can automatically produce derivatives given only a 
# # description of the model using the function tf.gradients. For simplicity, optimizers typically do this for you.

# # linear model
# W = tf.Variable([.3], dtype=tf.float32)
# b = tf.Variable([-.3], dtype=tf.float32)
# x = tf.placeholder(tf.float32)
# linear_model = W * x + b

# init = tf.global_variables_initializer()
# sess.run(init)

# y = tf.placeholder(tf.float32)
# # loss funtion
# squared_deltas = tf.square(linear_model - y)  
# loss = tf.reduce_sum(squared_deltas)

# optimizer = tf.train.GradientDescentOptimizer(0.01)
# train = optimizer.minimize(loss)

# sess.run(init) # reset values to incorrect defaults.
# for i in range(1000): # train 1000 times 
#   sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})

# print(sess.run([W, b])) # final value for W and b after training


import numpy as np
import tensorflow as tf

# Model parameters
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)
# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
# training data
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(1000):
  sess.run(train, {x:x_train, y:y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:x_train, y:y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))