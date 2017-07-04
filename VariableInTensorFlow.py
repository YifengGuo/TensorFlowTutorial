import tensorflow as tf 
# In machine learning we will typically want a model that can take arbitrary inputs, 
# such as the one above. To make the model trainable, we need to be able to modify the graph to get new 
# outputs with the same input. 
# Variables allow us to add trainable parameters to a graph. They are constructed with a type and initial value:

W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

sess = tf.Session()

# Constants are initialized when you call tf.constant, and their value can never change. 
# By contrast, variables are not initialized when you call tf.Variable. 

# To initialize all the variables in a TensorFlow program, you must explicitly call a special operation as follows:
init = tf.global_variables_initializer()
sess.run(init)


print(sess.run(linear_model, {x:[1,2,3,4]})) # Since x is a placeholder, we can evaluate linear_model for several values of x simultaneously

# loss function: A loss function measures how far apart the current model is from the provided data. 
y = tf.placeholder(tf.float32)
# standard loss model for linear regression which sums the squares of the deltas between the current model and the provided data. 
# linear_model - y creates a vector where each element is the corresponding example's error delta. We call tf.square to square that error.  
# Then, we sum all the squared errors to create a single scalar that abstracts the error of all examples using tf.reduce_sum:
squared_deltas = tf.square(linear_model - y)  
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]})) # 23.66

# to manually reassign the values of W and b(Variable can be reassigned by tf.assign()) 
fixW = tf.assign(W,[-1.])
fixb = tf.assign(b,[1.])
sess.run([fixW,fixb])
print(sess.run(loss,{x:[1,2,3,4], y:[0,-1,-2,-3]})) # 0.0

