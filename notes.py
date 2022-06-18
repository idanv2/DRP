import tensorflow as tf
import numpy as np
filter_x_l=tf.Variable(tf.constant([-1.,1.],shape=[2,1,1]))
E=tf.Variable(tf.constant([[1.,1.,1.],[2.,2.,2.],[3.,3.,3.]],shape=[3,3,1]))
q=tf.nn.conv1d(tf.transpose(E,perm=[1, 0, 2]),filter_x_l,stride=1, padding='SAME')






class coding_values:
    def __init__(self, n,k_size):
        self.Erow=(k_size,n - k_size)
        self.Ecol = (k_size, n - k_size)
        self.Hxrow=(k_size,n - k_size)
        self.Hxcol=(k_size - 1,n - k_size)
        self.Hycol = (k_size, n - k_size)
        self.Hyrow = (k_size - 1, n - k_size)
        E = np.zeros((n, n));  E[self.Erow[0]:self.Erow[1],self.Ecol[0]:self.Ecol[1]]=1
        Hx = np.zeros((n, n)); Hx[self.Hxrow[0]:self.Hxrow[1],self.Hxcol[0]:self.Hxcol[1]]=1
        Hy = np.zeros((n, n)); Hy[self.Hyrow[0]:self.Hyrow[1],self.Hycol[0]:self.Hycol[1]]=1
        self.E=tf.constant(E,shape=(n,n,1)); self.Hx=tf.constant(Hx,shape=(n,n,1)); self.Hy=tf.constant(Hy,shape=(n,n,1))
class BC_L:
    def __init__(self,E,Hx,Hy):
        E=E.numpy(); E[:,0]=1;E[0,:]=1;E[:,-1]=1;E[-1,:]=1
        self.Ebc=(tf.constant(E)+1)%2
        Hx =Hx.numpy();Hx[0,:]=1;Hx[-1,:]=1;Hx[:,-1]=1
        self.Hxbc = (tf.constant(Hx) + 1) % 2
        Hy = Hy.numpy();
        Hy[:, 0] = 1;Hy[:, -1] = 1;Hy[-1, :] = 1
        self.Hybc = (tf.constant(Hy) + 1) % 2

#





#
#print(E)
# input = tf.Variable(tf.constant(1.0, shape=[5, 5, 1]))
# #out_channels = 1
# filter = tf.Variable(tf.constant([-1.0, 0], shape=[2, 1, 1]))
# op = tf.nn.conv1d(input, filter, stride=1, padding='SAME')
