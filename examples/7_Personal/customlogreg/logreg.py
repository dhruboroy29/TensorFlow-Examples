from __future__ import print_function
import numpy as np, pandas as pd
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import tensorflow as tf
from sklearn.model_selection import KFold


def logreg_nomad(train_filename, test_filenames, beta, learning_rate, training_epochs, display_step):
    # Process .arff file
    dataset = arff.loadarff(train_filename)
    df = pd.DataFrame(dataset[0])

    # Extract data and labels
    data = df.select_dtypes(exclude=[object]).as_matrix()
    labels = df.select_dtypes(include=[object])

    num_data = data.shape[0] # unused
    num_features = data.shape[1]
    num_labels = labels.size # unused
    num_classes = np.unique(labels).size

    # Encode labels
    enc = LabelEncoder()
    encoded_labels = labels.apply(enc.fit_transform)

    # Convert labels to one-hot
    onehot = OneHotEncoder()
    onehot.fit(encoded_labels)
    onehot_labels = onehot.transform(encoded_labels).toarray()

    # Initialize placeholders
    x = tf.placeholder(tf.float32, [None, num_features])
    y = tf.placeholder(tf.float32, [None, num_classes])

    # Set model weights
    W = tf.Variable(tf.zeros([num_features, num_classes]))
    b = tf.Variable(tf.zeros([num_classes]))

    # Construct model
    logits = tf.matmul(x, W) + b  # Logits
    pred = tf.nn.softmax(logits)  # Softmax
    regularizer = tf.nn.l2_loss(W)  # l2 regularization

    # Minimize error using cross entropy
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y) + beta * regularizer)

    # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Initialize csv_lists
    csv_orig = [train_filename, beta, learning_rate]
    csv_10fold = [train_filename, beta, learning_rate]

    # Start training
    with tf.Session() as sess:
        '''
        10fold CrossValidation
        '''
        # Perform 10-fold split
        kf = KFold(n_splits=10, shuffle=True)

        for train_index, test_index in kf.split(data):
            # print("Train: ", train_index, "Test: ", test_index)

            # Re-run the initializer
            sess.run(init)

            X_train, X_test = data[train_index], data[test_index]
            y_train, y_test = onehot_labels[train_index], onehot_labels[test_index]

            # Print starting cost
            init_cost = sess.run(cost, feed_dict={x: X_train, y: y_train})
            print("Epoch:", '%04d' % 0, "cost=", "{:.9f}".format(init_cost))

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
            csv_10fold.append(acc_round)

        '''
        Train original file
        '''
        # Print starting cost
        init_cost = sess.run(cost, feed_dict={x: data, y: onehot_labels})
        print("Epoch:", '%04d' % 0, "cost=", "{:.9f}".format(init_cost))

        for epoch in range(training_epochs):
            # Run optimization
            _, c = sess.run([optimizer, cost], feed_dict={x: data, y: onehot_labels})
            if (epoch + 1) % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))

        # Print cost
        print('Optimization finished! Cost=%s' % c)

        '''Get precision for test files'''


