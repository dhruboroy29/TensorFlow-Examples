import tensorflow as tf

# Initialize weight variable (trainable)
W = tf.Variable([0.3], dtype=tf.float32)

# Initialize bias variable (trainable)
b = tf.Variable([-0.3], dtype=tf.float32)

# Initialize input placeholder
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# Define model
linear_model = W * x + b

# Start session
sess = tf.Session()

# Initialize global variables
init = tf.global_variables_initializer()
sess.run(init)

# Print heading for bad advanced linear_model model
print("#############################")
print("#Advanced Linear Model (bad)#")
print("#############################")

# Print weight and bias
print("[W], [b]: ", W.eval(session=sess), b.eval(session=sess))

# Now, define error function
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

# Print loss given input x and y
print("Loss: ", sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}), "\n")

# Print heading for best advanced linear_model model
print("##############################")
print("#Advanced Linear Model (best)#")
print("##############################")

# Reassign W and b
fixW = tf.assign(W, [-1])
fixb = tf.assign(b, [1])
print ("Reassigning [W], [b]: ", sess.run([fixW, fixb]))
print("Loss: ", sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}), "\n")