import tensorflow as tf 
node1 = tf.constant(3.0, dtype = tf.float32)
node2 = tf.constant(4.0)
print(node1,node2) # do not print direct value of node1 and node2

# To actually evaluate the nodes, we must run the computational graph within a session. 
# A session encapsulates the control and state of the TensorFlow runtime.
sess = tf.Session()
print(sess.run([node1,node2]))

# more complicated
node3 = tf.add(node1,node2)
print("node3: ", node3)
print("sess.run(node3)", sess.run(node3))