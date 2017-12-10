"""
Logistic regression on radar datasets implemented using loss and regularization formulae from Andrew Ng's site
(http://www.ritchieng.com/machine-learning/deep-learning/tensorflow/regularization/)
"""

from __future__ import print_function

import numpy as np, pandas as pd
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import tensorflow as tf
import datetime
from sklearn.model_selection import KFold

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

# Training epoch, display step
training_epochs = 2000
display_step = 100

# Learning rate, beta
learning_rate = 0.5
beta = 0.0001

# Initialize placeholders
x = tf.placeholder(tf.float32, [None,num_features])
y = tf.placeholder(tf.float32,[None,num_classes])

# Set model weights
W = tf.Variable(tf.zeros([num_features,num_classes]))
b = tf.Variable(tf.zeros([num_classes]))

# Construct model
logits = tf.matmul(x, W) + b # Logits
pred = tf.nn.softmax(logits)  # Softmax
regularizer = tf.nn.l2_loss(W) # l2 regularization

# Minimize error using cross entropy
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y)+beta*regularizer)

# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Test model
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # First, show results without 10-fold crossvalidation
    print('##########################################')
    print('#Logistic Regression - No Crossvalidation#')
    print('##########################################')

    # Print starting cost
    init_cost = sess.run(cost, feed_dict={x: data, y: onehot_labels})
    print("Epoch:", '%04d' % 0, "cost=", "{:.9f}".format(init_cost))

    t1 = datetime.datetime.now()
    for epoch in range(training_epochs):
        # Run optimization
        _, c = sess.run([optimizer, cost], feed_dict={x: data, y: onehot_labels})
        if (epoch + 1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))
    t2 = datetime.datetime.now()

    # Print cost
    print('Optimization finished! Cost=%s' % c)

    # Print runtime
    print('Optimization running time: %ss' % (t2-t1))

    # Print overall accuracy
    print("Accuracy: ", accuracy.eval({x: data, y: onehot_labels}),'\n')

    #--------------------------------------------------------------------------------------------------------------#

    # Next, show results with 10-fold crossvalidation
    print('###############################################')
    print('#Logistic Regression - 10-fold Crossvalidation#')
    print('###############################################')

    # Perform 10-fold split
    kf = KFold(n_splits=10, shuffle=True)

    # 10-fold crossvalidation accuracy
    acc_cv = 0

    # Initialize csv list
    csv_data = []
    csv_list = [beta, learning_rate]

    t3 = datetime.datetime.now()
    for train_index, test_index in kf.split(data):
        # print("Train: ", train_index, "Test: ", test_index)

        # Re-run the initializer
        sess.run(init)

        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = onehot_labels[train_index], onehot_labels[test_index]

        # Print starting cost
        init_cost = sess.run(cost, feed_dict={x: X_train, y: y_train})
        print("Epoch:", '%04d' % 0 , "cost=", "{:.9f}".format(init_cost))

        # Train for each partition
        for epoch in range(training_epochs):
            # Run optimization
            _, c = sess.run([optimizer, cost], feed_dict={x: X_train, y: y_train})
            if (epoch + 1) % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))

        # Print cost
        print('Optimization finished! Cost=%s' % c)

        # Now test corresponding test partition
        acc_round = accuracy.eval({x: X_test, y: y_test})
        print("Round accuracy: ", acc_round, '\n')
        acc_cv+= acc_round
        csv_list.append(acc_round)
    t4 = datetime.datetime.now()

    # Append row to csv_data
    csv_data.append(csv_list)

    # Print runtime
    print('Optimization running time: %ss' % (t4 - t3))

    # Print 10-fold crossvalidation accuracy
    print("10-fold cross-validation accuracy: ", acc_cv/10, '\n')

    # Save cross-validation output
    np.savetxt("10fold.csv", csv_data, delimiter=',')