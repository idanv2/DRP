import pickle

import tensorflow as tf
import numpy as np

from constants import Constants
from data_generator import generate_data
#ds
# Open the file in binary mode
with open('files/ex.pkl', 'rb') as file:
    ex = pickle.load(file)
with open('files/ey.pkl', 'rb') as file:
    ey = pickle.load(file)
with open('files/hx_x.pkl', 'rb') as file:
    hx_x = pickle.load(file)
with open('files/hy_x.pkl', 'rb') as file:
    hy_x = pickle.load(file)
with open('files/hx_y.pkl', 'rb') as file:
    hx_y = pickle.load(file)
with open('files/hy_y.pkl', 'rb') as file:
    hy_y = pickle.load(file)



import pickle

import tensorflow as tf
import numpy as np






def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(40,40)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(5)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=tf.keras.metrics.MeanSquaredError(),
    )
    return model


def train_model(model, x_train,y_train):
    model.fit(
        x_train,
        y_train,
        batch_size=10,
        epochs=3
        # validation_data=(x_val, y_val),
    )


#if __name__ == "__main__":
   # model = build_model()
    #train_model(model, np.random.rand(100,40,40),np.random.rand(100,5))

print(hx_x.shape)

