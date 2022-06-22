import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import notes
import pickle
from tensorflow import keras
from functions_for_torch import *




class par:
    lr = 1e-3;
    batch_size = 10;
    frog = 2;
    frog1 = 2
    nx = 30;
    ny = 30;
    C = notes.coding_values(nx, frog)
    E_code=C.E;Hx_code=C.Hx;Hy_code=C.Hy
    BC = notes.BC_L(C.E, C.Hx, C.Hy)
    Ebc = BC.Ebc; Hxbc=BC.Hxbc;Hybc=BC.Hybc
    ymin, ymax = 0.0, 1.0;
    xmin, xmax = 0.0, 1.0;
    Z = 1
    T = 0.1;
    time_steps = 400;
    dt = T / time_steps
    lx = xmax - xmin;
    ly = ymax - ymin;
    dx = lx / (nx - 1);
    dy = ly / (ny - 1)
    w_yee = tf.Variable(tf.constant([-1.,1.],shape=[2,1,1]))
    bc_left = tf.Variable(tf.constant([0,-1.,1.,0],shape=[4,1,1],dtype=tf.float64))


class Network(par):

    def __init__(self, w1):
        # super(Network,self).__init__()
        self.params = []
        self.params.append(w1)
        self.optimizer = keras.optimizers.SGD(learning_rate=1e-3)
        self.filter=self.params[0]
        self.rowpad=tf.constant([[0,0],[par.frog,par.frog],[0,0]],shape=[3,2])
        self.colpad=tf.constant([[par.frog,par.frog,0],[0,0,0]],shape=[3,2])
        self.rowpadE = tf.constant([[0, 0], [1, 1], [0, 0]], shape=[3, 2])
        self.colpadE = tf.constant([[1, 1, 0], [0, 0, 0]], shape=[3, 2])

    def forward(self, E,Hx,Hy):
        E= self.amper(E,Hx,Hy)
        Hx, Hy=self.faraday(E,Hx,Hy)
        return [E, Hx, Hy]

    def amper(self, E, Hx, Hy):
        S1 = tf.pad((par.Z * par.dt / par.dx) * self.Dx(Hy),self.colpad)
        S2 = tf.pad((par.Z * par.dt / par.dy) * self.Dy(Hx),self.rowpad)
        return (E + S1 - S2)

    def faraday(self, E,Hx,Hy):
        S3 = tf.pad((par.dt / (par.Z * par.dy)) * self.Dy(E),self.rowpadE)
        S4 = tf.pad((par.dt / (par.Z * par.dx)) * self.Dx(E),self.colpadE)
        Ax = (Hx- S3)
        Ay = (Hy+ S4)

        return Ax, Ay
    def Dy(self,B):
        return tf.nn.conv1d(B,self.filter,stride=1, padding='VALID')
    def Dx(self,B):
        return tf.transpose(tf.nn.conv1d(tf.transpose(B,perm=[1, 0, 2]), self.filter, stride=1, padding='VALID'),perm=[1, 0, 2])


w1 = tf.Variable(tf.constant([0,-1.,1.,0],shape=[4,1,1],dtype=tf.float64))
N=Network(w1)
E=tf.random.normal(shape=(par.nx,par.nx,1),dtype=tf.dtypes.float64)
Hx=tf.random.normal(shape=(par.nx,par.nx-1,1),dtype=tf.dtypes.float64)
Hy=tf.random.normal(shape=(par.nx-1,par.nx,1),dtype=tf.dtypes.float64)
#pad=tf.constant([[0,0],[2,1],[0,0]],shape=[3,2])
#pad=tf.constant([[2,1,0],[0,0,0]],shape=[3,2])
#E=tf.pad(E,pad)
N.faraday(E,Hx,Hy)
#print(tf.squeeze(N.amper(E,Hx,Hy)))

