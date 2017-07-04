import tensorflow as tf 

# A graph can be parameterized to accept external inputs, known as placeholders. 
# A placeholder is a promise to provide a value later.
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b # + provides a shortcut for tf.add(a, b) # like a lambda function

# The preceding three lines are a bit like a function or a lambda in which we define two input parameters (a and b) and 
# then an operation on them. We can evaluate this graph with multiple inputs by using the feed_dict parameter to 
# specify Tensors that provide concrete values to these placeholders:
sess = tf.Session()

print(sess.run(adder_node,{a:3, b:4.5}))
print(sess.run(adder_node, {a: [1,3], b: [2, 4]}))

add_and_triple = adder_node * 3 # another lambda function
print(sess.run(add_and_triple, {a: 3, b : 4.5})) 