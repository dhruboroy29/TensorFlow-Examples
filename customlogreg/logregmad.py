'''
MAD types
---------
featuremad: MAD of features in composite file
featuremedianmad: MAD of feature median in single environments
infogainmad: MAD of feature infogain rankings in single environments
mrmrmad: MAD of feature mrmr rankings in single environments
'''

from __future__ import print_function
import numpy as np, pandas as pd
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import tensorflow as tf
from sklearn.model_selection import KFold
import os
import datetime


def get_logreg_coeffs(train_filename, beta, training_epochs, learning_rate):
    # Process .arff file
    dataset = arff.loadarff(train_filename)
    df = pd.DataFrame(dataset[0])

    # Extract data and labels
    data = df.select_dtypes(exclude=[object]).as_matrix()
    labels = df.select_dtypes(include=[object])

    num_data = data.shape[0]  # unused
    num_features = data.shape[1]
    num_labels = labels.size  # unused
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
    regularizer = tf.nn.l2_loss(W)  # l2 regularization

    # Minimize error using cross entropy
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y) + beta * regularizer)

    # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Start training
    with tf.Session() as sess1:
        # Run the initializer
        sess1.run(init)

        for epoch in range(training_epochs):
            # Run optimization
            _, c, w = sess1.run([optimizer, cost, W], feed_dict={x: data, y: onehot_labels})

    # Return final weight
    return w


def compute_logregmad(single_env_files, num_features, beta, training_epochs, learning_rate): # TODO: Generalize for any number of classes
    count = 0
    for cur_file in single_env_files:
        f_weight = get_logreg_coeffs(cur_file, beta, training_epochs, learning_rate)
        if count==0:
            f_w1 = f_weight[:,0]
            f_w2 = f_weight[:,1]
        else:
            f_w1 = np.vstack((f_w1, f_weight[:,0])) # class1
            f_w2 = np.vstack((f_w2, f_weight[:,1])) # class2

        count += 1

    # Calculate mad
    mad1 = np.median(abs(f_w1 - np.median(f_w1, 0)), 0) # class1
    mad2 = np.median(abs(f_w2 - np.median(f_w2, 0)), 0)  # class2

    mad = np.vstack((mad1, mad2)) # Return MAD matrix

    # Normalize and return
    max_m = np.max(mad,1)
    min_m = np.min(mad,1)

    max_m = max_m.reshape([max_m.size, 1])
    min_m = min_m.reshape([min_m.size, 1])
    return (mad-min_m)/(max_m-min_m)


def compute_featuremad(mat):
    # Compute MAD
    med = np.median(mat, 0)
    mad = np.median(abs(mat-med),0)

    # Normalize and return
    # return (mad-min_x(mad))/(max_x(mad)-min_x(mad))
    return mad


def get_inversemad(madfile):
    mad = np.loadtxt(madfile, delimiter=',')
    mad[mad==0]=1e-20 # eps masking to avoid divide-by-zero error
    return 1.0/mad


def logreg_mad(train_filename, test_filenames, mad_file_or_single_envs, mad_type, beta, learning_rate, training_epochs, display_step, rescale=True):
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

    # Get MAD
    if mad_type=='featuremad':
        norm_mad = compute_featuremad(data)
    elif mad_type=='logregmad':
        norm_mad = compute_logregmad(mad_file_or_single_envs, num_features, beta, training_epochs, learning_rate)
    else:
        norm_mad = get_inversemad(mad_file_or_single_envs) # W*1/mad

    # Initialize tensorflow constant
    if mad_type !='logregmad':
        m = tf.constant(norm_mad, dtype=tf.float32, shape=[norm_mad.size,1])
    else:
        norm_mad_transp = np.transpose(norm_mad)
        m = tf.constant(norm_mad_transp, dtype=tf.float32)

    # Initialize placeholders
    x = tf.placeholder(tf.float32, [None, num_features])
    y = tf.placeholder(tf.float32, [None, num_classes])

    # Set model weights
    W = tf.Variable(tf.ones([num_features, num_classes]))
    b = tf.Variable(tf.zeros([num_classes]))

    # Construct model
    logits = tf.matmul(x, W) + b  # Logits
    pred = tf.nn.softmax(logits)  # Softmax

    # Multiply weights with MAD and perform min-max scaling
    if rescale:
        scaled_weights = tf.multiply(W, m)
        max_sc = tf.reduce_max(scaled_weights)
        min_sc = -1 * tf.reduce_max(-1 * scaled_weights)
        scaled_weights_01 = (scaled_weights-min_sc)/(max_sc - min_sc)
        regularizer = tf.nn.l2_loss(scaled_weights_01)  # l2 regularization with rescaling
    else:
        regularizer = tf.nn.l2_loss(tf.multiply(W, m))  # l2 regularization

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


def run_logreg_featuremad(base_path, round, beta_list, learning_rate, training_epochs, display_step):
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
            csv_train_cur, csv_test_cur = logreg_mad(tr_file[0],tst_files, None, 'featuremad', beta, learning_rate, training_epochs, display_step)
            csv_train.append(csv_train_cur)
            csv_test = csv_test + csv_test_cur # Merge instead of append, to reduce extra redundant dimension

    np.savetxt(os.path.join(round_dir,'training_featuremad.csv'), csv_train, fmt='%s', delimiter=',')
    np.savetxt(os.path.join(round_dir, 'testing_featuremad.csv'), csv_test, fmt='%s', delimiter=',')


def run_logreg_logregmad(base_path, round, beta_list, learning_rate, training_epochs, display_step):
    round_dir = os.path.join(base_path,'Round'+str(round))
    training_dirs = [name for name in os.listdir(round_dir) if os.path.isdir(os.path.join(round_dir, name))]
    training_dirs.sort()

    csv_train = []
    csv_test = []
    for tr in training_dirs:
        cur_training_dir = os.path.join(round_dir,tr,'combined')
        cur_test_dir = os.path.join(round_dir,tr,'test')
        cur_single_envs_dir = os.path.join(round_dir, tr, 'single_envs')

        # Get training file
        tr_file = [os.path.join(cur_training_dir, f) for f in os.listdir(cur_training_dir) if f.endswith('.arff')]
        # print('Training: ', tr_file)

        # Get test files
        tst_files = [os.path.join(cur_test_dir, f) for f in os.listdir(cur_test_dir) if f.endswith('.arff')]
        tst_files.sort()
        # print('Testing: ', tst_files)

        # Get single environments
        single_env_files = [os.path.join(cur_single_envs_dir, f) for f in os.listdir(cur_single_envs_dir) if f.endswith('.arff')]
        single_env_files.sort()

        for beta in beta_list:
            csv_train_cur, csv_test_cur = logreg_mad(tr_file[0],tst_files, single_env_files, 'logregmad', beta, learning_rate, training_epochs, display_step, rescale=False)
            csv_train.append(csv_train_cur)
            csv_test = csv_test + csv_test_cur # Merge instead of append, to reduce extra redundant dimension

    np.savetxt(os.path.join(round_dir,'training_logregmad_madscaled.csv'), csv_train, fmt='%s', delimiter=',')
    np.savetxt(os.path.join(round_dir, 'testing_logregmad_madscaled.csv'), csv_test, fmt='%s', delimiter=',')


def run_logreg_infogainmad(base_path, round, beta_list, learning_rate, training_epochs, display_step):
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

        # Get mad file (precomputed)
        mad_file = os.path.join(round_dir, tr, 'InfoGainMAD.txt')

        for beta in beta_list:
            csv_train_cur, csv_test_cur = logreg_mad(tr_file[0],tst_files, mad_file, 'infogainmad', beta, learning_rate, training_epochs, display_step)
            csv_train.append(csv_train_cur)
            csv_test = csv_test + csv_test_cur # Merge instead of append, to reduce extra redundant dimension

    np.savetxt(os.path.join(round_dir,'training_infogainmad.csv'), csv_train, fmt='%s', delimiter=',')
    np.savetxt(os.path.join(round_dir, 'testing_infogainmad.csv'), csv_test, fmt='%s', delimiter=',')


def run_logreg_mrmrmad(base_path, round, beta_list, learning_rate, training_epochs, display_step):
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

        # Get mad file (precomputed)
        mad_file = os.path.join(round_dir, tr, 'mRMRMAD.txt')

        for beta in beta_list:
            csv_train_cur, csv_test_cur = logreg_mad(tr_file[0],tst_files, mad_file, 'mrmrmad', beta, learning_rate, training_epochs, display_step)
            csv_train.append(csv_train_cur)
            csv_test = csv_test + csv_test_cur # Merge instead of append, to reduce extra redundant dimension

    np.savetxt(os.path.join(round_dir,'training_mrmrmad.csv'), csv_train, fmt='%s', delimiter=',')
    np.savetxt(os.path.join(round_dir, 'testing_mrmrmad.csv'), csv_test, fmt='%s', delimiter=',')


if __name__ == '__main__':
    base_path = '/Users/Balderdash/Documents/BigEnvs_LogReg/BigEnvs_LogReg'
    rounds = [2, 3, 4, 5]
    beta_list = [0.1, 0.01, 0.001, 0.0001]
    epochs = 2000
    step = 100
    rate = 0.5

    for round in rounds:
        t1 = datetime.datetime.now()
        run_logreg_logregmad(base_path, round, beta_list, learning_rate=rate, training_epochs=epochs, display_step=step)
        t2 = datetime.datetime.now()
        # Print runtime
        print('Net running time: %s' % (t2 - t1))
