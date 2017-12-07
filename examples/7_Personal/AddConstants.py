import tensorflow as tf

# Create constant node with explicit type
node1 = tf.constant(3.0, dtype=tf.float32)

# Create constant node with implicit type
node2 = tf.constant(4.0)

# Create addition node
node_add = tf.add(node1, node2)

# Start new session
sess = tf.Session()

# Evaluate nodes
print("[node1, node2]: ", sess.run([node1, node2]))

#Evaluate sum
print("sum: ", sess.run(node_add))