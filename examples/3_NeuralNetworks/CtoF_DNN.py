from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np

l0 = Dense(1, input_shape=[1])
m = Sequential([l0])

m.compile(loss='mse', optimizer=Adam(0.1))

celsius_q    = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)

m.summary()

hist = m.fit(celsius_q, fahrenheit_a, epochs=10000, verbose=False)

print("Learned weights: {}".format(m.get_weights()))

print("{} degrees C = {} degrees F".format(0, m.predict([0])))

# Plot loss function
import matplotlib.pyplot as plt
plt.plot(hist.history['loss'])
plt.xlabel('Epoch#')
plt.ylabel('Loss')
plt.show()