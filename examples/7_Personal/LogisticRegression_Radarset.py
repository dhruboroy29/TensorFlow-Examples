from __future__ import print_function

import numpy as np, pandas as pd
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import tensorflow as tf
import datetime

# Process .arff file
dataset = arff.loadarff('radar1_scaled.arff')
df = pd.DataFrame(dataset[0])

# Extract data and labels
data = df.select_dtypes(exclude=[object]).as_matrix()
labels = df.select_dtypes(include=[object])

num_data = data.shape[0]
num_features = data.shape[1]
num_labels = labels.size
num_classes = np.unique(labels).size

# print('Num data = %s, num labels = %s, num features = %s' % (num_data, num_labels, num_features))

# Encode labels
enc = LabelEncoder()
encoded_labels = labels.apply(enc.fit_transform)

# Convert labels to one-hot
onehot = OneHotEncoder()
onehot.fit(encoded_labels)
onehot_labels = onehot.transform(encoded_labels).toarray()

# print("One-Hot labels: %s\n" % onehot_labels)

# Learning rate
learning_rate = 0.01

# Initialize placeholders
x = tf.placeholder(tf.float32, [None,num_features])
y = tf.placeholder(tf.float32,[None,num_classes])

# Set model weights
W = tf.Variable(tf.zeros([num_features,num_classes]))
b = tf.Variable(tf.zeros([num_classes]))

# Construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b)  # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))

# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

training_epochs = 1000000

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    t1 = datetime.datetime.now()
    for epoch in range(training_epochs):
        # Run optimization
        _, c = sess.run([optimizer, cost], feed_dict={x: data, y: onehot_labels})
        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))
    t2 = datetime.datetime.now()

    # Print cost
    print('Optimization finished! Cost=%s' % c)

    # Print runtime
    print('Optimization running time: %ss' % (t2-t1))

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy: ", accuracy.eval({x: data, y: onehot_labels}))