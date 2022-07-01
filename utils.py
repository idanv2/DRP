import numpy as np
import tensorflow as tf
from tensorflow import keras

from constants import Constants


class Network:

    def __init__(self, w):
        # super(Network,self).__init__()
        self.params = w
        self.optimizer = keras.optimizers.SGD(learning_rate=1e-3)
        self.filter = tf.Variable(
            tf.constant([[0, -1., 1., 0], [0, -1., 1., 0], [0, -1., 1., 0]], shape=[3, 4, 1, 1], dtype=tf.float64))
        self.forward = tf.Variable(
            tf.constant([0, -1., 1., 0], shape=[1, 4, 1, 1], dtype=tf.float64), trainable=False)
        self.backward = tf.Variable(
            tf.constant([0, -1., 1., 0], shape=[1, 4, 1, 1], dtype=tf.float64), trainable=False)
        self.pad1=tf.constant([[0, 0], [2, 2], [2, 2], [0,0]], shape=[4, 2])
        self.pad2=tf.constant([[0, 0], [0, 0], [1, 1], [0, 0]], shape=[4, 2])
        self.pad3=tf.constant([[0, 0], [1, 1], [0, 0], [0, 0]], shape=[4, 2])

    def forward(self, E, Hx, Hy):
        E = self.amper(E, Hx, Hy)
        Hx, Hy = self.faraday(E, Hx, Hy)
        return [E, Hx, Hy]

    def amper(self, E, Hx, Hy):
        S1 = tf.pad(( Constants.DT / Constants.DX) * self.Dx(Hy,tf.transpose(self.filter, perm=[1, 0, 2, 3])), self.pad1)+ \
        tf.pad(self.Dx(Hy,tf.transpose(Constants.KERNEL_FORWARD, perm=[1, 0, 2, 3])),Constants.PADY_FORWARD)+ \
        tf.pad(self.Dx(Hy, tf.transpose(Constants.KERNEL_BACKWARD, perm=[1, 0, 2, 3])), Constants.PADY_BACWARD)

        S2 = tf.pad((Constants.Z * Constants.DT / Constants.DY) * self.Dy(Hx,self.filter), self.pad1)+ \
        tf.pad(self.Dy(Hx, Constants.KERNEL_FORWARD), Constants.PADX_FORWARD)+ \
        tf.pad(self.Dy(Hx, Constants.KERNEL_BACKWARD), Constants.PADX_BACWARD)
        return (E + S1 - S2)

    def faraday(self, E, Hx, Hy):
        S3 = (Constants.DT / (Constants.Z * Constants.DY))*tf.pad(self.Dy(E,self.filter), self.pad2)+ \
        tf.pad(self.Dy(E, tf.transpose(Constants.KERNEL_E_FORWARD, perm=[1, 0, 2, 3]))[:,1:-1,:,:], Constants.PADEX_FORWARD) + \
        tf.pad(self.Dx(E, tf.transpose(Constants.KERNEL_E_BACKWARD, perm=[1, 0, 2, 3]))[:,1:-1,:,:], Constants.PADEY_BACKWARD)

        S4 = (Constants.DT / (Constants.Z * Constants.DX))*tf.pad(self.Dx(E,tf.transpose(self.filter, perm=[1, 0, 2, 3])), self.pad3)+ \
        tf.pad(self.Dx(E, tf.transpose(Constants.KERNEL_E_FORWARD, perm=[1, 0, 2, 3])), Constants.PADEY_FORWARD) + \
        tf.pad(self.Dx(E, tf.transpose(Constants.KERNEL_E_BACKWARD, perm=[1, 0, 2, 3])), Constants.PADEY_BACKWARD)

        Ax = (Hx - S3)
        Ay = (Hy + S4)

        return Ax, Ay


    def Dy(self, B, kernel ):
        return tf.nn.conv2d(B, kernel, stride=1, padding='VALID')


    def Dx(self, B, kernel):
        return  tf.nn.conv2d(B, kernel, stride=1, padding='VALID')



def f_a(c, n, k1, k2):
    e = c * np.cos(c * n * Constants.DT) * (
            np.sin(Constants.PI * k1 * Constants.X) * np.sin(Constants.PI * k2 * Constants.Y) +
            np.sin(Constants.PI * k2 * Constants.X) * np.sin(
        Constants.PI * k1 * Constants.Y))

    hx = np.sin(c * (Constants.DT / 2) * (2 * n + 1)) * (
            -Constants.PI * k2 * np.sin(Constants.PI * k1 * Constants.X) * np.cos(
        Constants.PI * k2 * (Constants.Y + Constants.DX / 2)) - Constants.PI * k1 * np.sin(
        Constants.PI * k2 * Constants.X) * np.cos(Constants.PI * k1 * (Constants.Y + Constants.DX / 2)))

    hy = np.sin(c * (Constants.DT / 2) * (2 * n + 1)) * (
            Constants.PI * k1 * np.cos(Constants.PI * k1 * (Constants.X + Constants.DX / 2)) * np.sin(
        Constants.PI * k2 * Constants.Y) + Constants.PI * k2 * np.cos(
        Constants.PI * k2 * (Constants.X + Constants.DX / 2)) * np.sin(Constants.PI * k1 * Constants.Y))

    return e, hx[1:-1, :-1], hy[:-1, 1:-1]
