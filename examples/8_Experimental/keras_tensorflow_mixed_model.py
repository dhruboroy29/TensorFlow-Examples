import tensorflow as tf
import keras
from keras import backend as K

# Link Tensorflow session to Keras session
sess = tf.Session()
K.set_session(sess)

'''Create mixed network'''
# This placeholder will contain our input digits, as flat vectors
img = tf.placeholder(tf.float32, shape=(None, 784))

# Input Layer
input_layer = tf.reshape(img, [-1, 28, 28, 1])

# Convolutional Layer #1 (TensorFlow layer)
conv1 = tf.layers.conv2d(
    inputs=input_layer,
    filters=32,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu)

# Flatten conv output (TensorFlow layer)
flat = tf.layers.flatten(conv1)

# Fully-connected Layer #1 (Keras layer)
layer2_dense = keras.layers.Dense(128, activation='relu')(flat)

# Fully-connected Output Layer (TensorFlow layer) - DOES NOT WORK: WTF!
#output_preds = tf.layers.dense(inputs=layer2_dense, units=10, activation=tf.nn.softmax)

# Or, equivalently, fully-connected Output Layer (Keras layer)
output_preds = keras.layers.Dense(10, activation='softmax')(layer2_dense)

# True labels (TensorFlow layer
labels = tf.placeholder(tf.float32, shape=(None, 10))

# Loss model from Keras
from keras.objectives import categorical_crossentropy
loss = tf.reduce_mean(categorical_crossentropy(labels, output_preds))


'''Load lata'''
from tensorflow.examples.tutorials.mnist import input_data
mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)


'''Train network - TensorFlow'''
# Training step
train_step = tf.train.AdamOptimizer(0.5).minimize(loss)

#TODO: Implement network training alternative in Keras

# Initialize all variables
init_op = tf.global_variables_initializer()
sess.run(init_op)

# Run training loop
with sess.as_default():
    for i in range(10000):
        batch = mnist_data.train.next_batch(50)
        train_step.run(feed_dict={img: batch[0], labels: batch[1]})


'''Evaluate model - Keras'''
from keras.metrics import categorical_accuracy as accuracy
acc_value = tf.reduce_mean(accuracy(labels, output_preds))
with sess.as_default():
    print("Accuracy: ",acc_value.eval(feed_dict={img: mnist_data.test.images,
                                    labels: mnist_data.test.labels}))



