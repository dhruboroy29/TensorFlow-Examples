from __future__ import print_function
import numpy as np, pandas as pd
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import tensorflow as tf
from sklearn.model_selection import KFold
import os
import datetime


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
    W = tf.Variable(tf.ones([num_features, num_classes]))
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
    csv_10foldtrain = [train_filename, beta, learning_rate]
    csv_10foldtest = []

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
            csv_10foldtrain.append(acc_round)

        '''
        Train original file
        '''
        # Run the initializer
        sess.run(init)

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
        # Idea: convert to one-hot, extract humans labels and corresponding features, run accuracy
        for cur_testfile in test_filenames:
            cur_list = [train_filename, cur_testfile, beta, learning_rate];

            # Process .arff file
            testdataset = arff.loadarff(cur_testfile)
            df = pd.DataFrame(testdataset[0])

            # Extract data and labels
            testdata = df.select_dtypes(exclude=[object]).as_matrix()
            testlabels = df.select_dtypes(include=[object])

            # Encode labels
            enc = LabelEncoder()
            encoded_testlabels = testlabels.apply(enc.fit_transform)

            # Convert labels to one-hot
            onehot = OneHotEncoder()
            onehot.fit(encoded_testlabels)
            onehot_testlabels = onehot.transform(encoded_testlabels).toarray()

            # Extract humans
            humans = onehot_testlabels[:,1]==1 # Only valid for Human-Dog radar datasets, since humans are 0 1 in one-hot
            onehot_testlabels_humans = onehot_testlabels[humans]
            testdata_humans = testdata[humans]

            precision = accuracy.eval({x: testdata_humans, y: onehot_testlabels_humans})
            cur_list.append(precision)

            # Append to main list
            csv_10foldtest.append(cur_list)

    # Return csv lists
    return csv_10foldtrain, csv_10foldtest


def run_logreg(base_path, round, beta_list, learning_rate, training_epochs, display_step):
    round_dir = os.path.join(base_path,'Round'+str(round))
    training_dirs = [name for name in os.listdir(round_dir) if os.path.isdir(os.path.join(round_dir, name))]
    training_dirs.sort()

    csv_train = []
    csv_test = []
    for tr in training_dirs:
        cur_training_dir = os.path.join(round_dir,tr,'combined')
        cur_test_dir = os.path.join(round_dir,tr,'test')

        # Get training file
        tr_file = [os.path.join(cur_training_dir, f) for f in os.listdir(cur_training_dir) if f.endswith('.arff')]
        # print('Training: ', tr_file)

        # Get test files
        tst_files = [os.path.join(cur_test_dir, f) for f in os.listdir(cur_test_dir) if f.endswith('.arff')]
        tst_files.sort()
        # print('Testing: ', tst_files)

        for beta in beta_list:
            csv_train_cur, csv_test_cur = logreg_nomad(tr_file[0],tst_files,beta,learning_rate,training_epochs,display_step)
            csv_train.append(csv_train_cur)
            csv_test = csv_test + csv_test_cur # Merge instead of append, to reduce extra redundant dimension

    np.savetxt(os.path.join(round_dir,'training.csv'), csv_train, fmt='%s', delimiter=',')
    np.savetxt(os.path.join(round_dir, 'testing.csv'), csv_test, fmt='%s', delimiter=',')


if __name__ == '__main__':
    base_path = '/Users/Balderdash/Documents/BigEnvs_LogReg/BigEnvs_LogReg'
    rounds = [2, 3, 4, 5]
    beta_list = [0.1, 0.01, 0.001, 0.0001]
    epochs = 2000
    step = 100
    rate = 0.5

    for round in rounds:
        t1 = datetime.datetime.now()
        run_logreg(base_path, round, beta_list, learning_rate=rate, training_epochs=epochs, display_step=step)
        t2 = datetime.datetime.now()
        # Print runtime
        print('Net running time: %s' % (t2 - t1))
