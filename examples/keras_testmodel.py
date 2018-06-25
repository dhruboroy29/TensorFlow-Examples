from __future__ import print_function

import pickle
import numpy as np
import sys
import os

from keras.optimizers import Adam, SGD
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Reshape, Dropout  # InputLayer
# from keras.layers.recurrent import LSTM
from keras.layers.recurrent import GRU

# Input directory
in_path = sys.argv[1]
# Num classes
nb_classes = 10
np.random.seed(1337)  # for reproducibility
n_mels = 256
# Hidden units
hidden_units = int(sys.argv[2])  # 256
nb_epochs = int(sys.argv[3])  # 100
dropout_rate = float(sys.argv[4])  # 0.2
learning_rate = float(sys.argv[5])  # 1e-4
# Batch size
batch_size = int(sys.argv[6])  # 64
# Optimizer
if sys.argv[7].lower() == 'Adam'.lower():
    optimizer = Adam(lr=learning_rate)
elif sys.argv[7].lower() == 'SGD'.lower():
    optimizer = SGD(lr=learning_rate, momentum=0.9, nesterov=True)
# Stacked?
if len(sys.argv) > 8 and sys.argv[8].lower().__contains__('Stack'.lower()):  # Last argument can be stack/stacked/none
    stacked = True
else:
    stacked = False

# Dropout?
if dropout_rate == 0.0:
    dropout = True
else:
    dropout = False

# Output filenames
if stacked:
    history_fname = 'stackedgruhist_h=' + sys.argv[2] + '_e=' + sys.argv[3] + '_d=' + sys.argv[4] + '_l=' + sys.argv[
        5] + '_b=' + sys.argv[6] + '_' + sys.argv[7].lower() + '.pkl'
else:
    history_fname = 'gruhist_h=' + sys.argv[2] + '_e=' + sys.argv[3] + '_d=' + sys.argv[4] + '_l=' + sys.argv[
        5] + '_b=' + sys.argv[6] + '_' + sys.argv[7].lower() + '.pkl'

n_steps = 199

# Initialize model
model = Sequential()
model.add(Reshape((-1, n_mels), input_shape=(50944,)))
# model.add(LSTM(units=hidden_units, init='uniform', inner_init='uniform',
#            forget_bias_init='one', input_shape=(n_steps,n_mels)))
# model.add(LSTM(kernel_initializer="uniform", input_shape=(n_steps, n_mels),
#               recurrent_initializer="uniform", units=hidden_units, unit_forget_bias=True))
model.add(GRU(kernel_initializer="uniform", input_shape=(n_steps, n_mels),
              recurrent_initializer="uniform", units=hidden_units, return_sequences=True))
# Add dropout layer (optional)
if dropout:
    model.add(Dropout(rate=dropout_rate, seed=1337))
# Add stacked layer (optional)
if stacked:  # TODO: Make stacked hidden unit different
    model.add(GRU(kernel_initializer="uniform", recurrent_initializer="uniform", units=hidden_units))
    # Add second dropout layer (optional)
    if dropout:  # TODO: Make stacked dropout rate different
        model.add(Dropout(rate=dropout_rate, seed=1337))
model.add(Dense(nb_classes, activation='softmax'))

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()