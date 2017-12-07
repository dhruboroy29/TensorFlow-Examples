import tensorflow as tf
import time

# Fix number of iterations in optimization
num_iterations = 10000

# Model parameters
W = tf.Variable([.5], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)

# Model I/O
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# Linear model
linear_model = W * x + b;

# l2 loss function
loss = tf.reduce_sum(tf.square(linear_model - y))

# Optimizer
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate)

# Training
train = optimizer.minimize(loss)

# Input data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

# Session and global variable initializer
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # Set values to wrong

# Training loop with profiling
start_time = time.time()
for i in range(num_iterations):
    sess.run(train, {x: x_train, y: y_train})
end_time = time.time()

# Evaluate training accuracy
final_W, final_b, final_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("Final W: %s b: %s loss: %s" % (final_W, final_b, final_loss))
print("Elapsed time was %g seconds" % (end_time - start_time))