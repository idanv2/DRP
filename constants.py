import math

import numpy as np
from dataclasses import dataclass

import tensorflow as tf


@dataclass(frozen=True)
class Constants:
    LR = 1e-3
    BATCH_SIZE = 10
    FROG = 2
    FROG1 = 2
    N = 30
    PI=math.pi
    YMIN, YMAX = 0.0, 1.0
    XMIN, XMAX = 0.0, 1.0
    Z = 1
    T = 0.1
    TIME_STEPS = 400
    DT = T / TIME_STEPS
    LX = XMAX-XMIN
    LY = YMAX-YMIN
    DX = LX / (N - 1)
    DY = LY / (N - 1)
    X, Y = np.meshgrid(np.linspace(0., XMAX, N), np.linspace(0., XMAX, N), indexing='ij')
    W_YEE = tf.Variable(tf.constant([-1., 1.], shape=[2, 1, 1]))
    BC_LEFT = tf.Variable(tf.constant([0, -1., 1., 0], shape=[4, 1, 1], dtype=tf.float64))

    PADX_FORWARD = tf.constant([[0, 0], [1, 1], [1, N-2], [0, 0]], shape=[4, 2])
    PADX_BACWARD = tf.constant([[0, 0], [1, 1], [N-2, 1], [0, 0]], shape=[4, 2])
    PADY_FORWARD = tf.constant([[0, 0], [1, N-2], [1, 1], [0, 0]], shape=[4, 2])
    PADY_BACWARD = tf.constant([[0, 0], [N-2, 1], [1, 1], [0, 0]], shape=[4, 2])

    A = np.array([1., 2., 3., 4., 5.]).reshape(1, 5)
    B = np.zeros((1, N-5-1))
    KERNEL_FORWARD = np.append(A, B).reshape(1, N-1, 1, 1)
    KERNEL_BACKWARD = np.append(A, B).reshape(1, N-1, 1, 1)

    PADEX_FORWARD=tf.constant([[0, 0], [0, 0], [0, N-2], [0, 0]], shape=[4, 2])
    PADEX_BACKWARD=tf.constant([[0, 0], [0, 0], [N-2, 0], [0, 0]], shape=[4, 2])
    PADEY_FORWARD = tf.constant([[0, 0], [0, N-2], [0, 0], [0, 0]], shape=[4, 2])
    PADEY_BACKWARD = tf.constant([[0, 0], [N-2, 0], [0, 0], [0, 0]], shape=[4, 2])

    C = np.array([1., 2., 3., 4., 5.]).reshape(1, 5)
    D = np.zeros((1, N-5))
    KERNEL_E_FORWARD = np.append(C, D).reshape(1, N, 1, 1)
    KERNEL_E_BACKWARD = np.append(C, D).reshape(1, N, 1, 1)


