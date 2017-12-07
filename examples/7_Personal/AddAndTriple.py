import tensorflow as tf

# Define placeholders
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# Create operation node
add_and_triple_node = (x + y) * 3.0 # Shortcut for tf.multiply(tf.add(x, y), tf.constant(3.0))

# Create operation node explicitly
add_and_triple_node_2 = tf.multiply(tf.add(x, y), tf.constant(3.0))

# Start new session
sess = tf.Session()

# Evaluate operation on scalars
print("Add-and-triple scalars: ", sess.run(add_and_triple_node, {x: 3, y: 4.5}))
print("Add-and-triple scalars 2: ", sess.run(add_and_triple_node_2, {x: 2.3, y: -4.5}))

# Evaluate operation on vectors
print("Add-and-triple vectors: ", sess.run(add_and_triple_node, {x: [1, 3], y: [2, 4]}))
print("Add-and-triple vectors 2: ", sess.run(add_and_triple_node_2, {x: [1, -3, 3], y: [2, 0.5, 4]}))

