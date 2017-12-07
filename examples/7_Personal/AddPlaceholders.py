import tensorflow as tf

# Define placeholders
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# Create adder node
adder_node = x + y # Shortcut for tf.add(x, y)

# Create adder node explicitly
adder_node_2 = tf.add(x, y)

# Start new session
sess = tf.Session()

# Evaluate sum of scalars
print("Sum of scalars: ", sess.run(adder_node, {x: 3, y: 4.5}))
print("Sum of scalars 2: ", sess.run(adder_node_2, {x: 2.3, y: -4.5}))

# Evaluate sum of vectors
print("Sum of vectors: ", sess.run(adder_node, {x: [1, 3], y: [2, 4]}))
print("Sum of vectors 2: ", sess.run(adder_node_2, {x: [1, -3, 3], y: [2, 0.5, 4]}))