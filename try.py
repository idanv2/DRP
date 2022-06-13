import pickle
import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from plot_graphs import generate_data
from main6 import run_p,Average
xmin, xmax = 0.0, 1.0  # limits in the x direction
ymin, ymax = 0.0, 1.0
# dt=0.0002
nx=40
ny=40
#k1=8.
#k2=8.
T=1
time_steps=80
dt = T / time_steps  # limits in the y direction
lx = xmax - xmin  # domain length in the x direction
ly = ymax - ymin  # domain length in the y direction
dx = lx / (nx - 1)  # grid spacing in the x direction
dy = ly / (ny - 1)  # grid spacing in the y direction


w_yee=torch.tensor([1.],dtype=float,requires_grad=False)
# Open the file in binary mode
with open('w1.pkl', 'rb') as file:
    # Call load method to deserialze
    w1 = pickle.load(file).detach().clone()
L=[3.,4.]

Loss=[]
Loss_yee=[]
for k1 in L:
    E_a, Hx_a, Hy_a = generate_data(1., nx, ny, k1, k1, dx=dx, dy=dy, dt=dt,
                                    time_steps=time_steps)
    Loss.append(run_p(w1, k1, k1, nx, ny, T, time_steps, E_a=E_a, Hx_a=Hx_a, Hy_a=Hy_a, dt=dt,
                        dx=dx, dy=dy,training=False))
    Loss_yee.append(run_p(w_yee, k1, k1, nx, ny, T, time_steps, E_a=E_a, Hx_a=Hx_a, Hy_a=Hy_a, dt=dt,
                        dx=dx, dy=dy,training=False))
plt.xlabel('k')
# Set the y axis label of the current axis.
plt.title('dx=dy='+str(1/nx)+', '+'dt='+str(dt)+', '+'Time steps='+str(time_steps))
plt.ylabel('error')
plt.plot(L,Loss, color='blue', linewidth = 3,  label = 'DL: 4-point stencil')
plt.plot(L,Loss_yee, color='red', linewidth = 3,  label = 'Yee scheme')
plt.plot(L,[0]*len(L), color='black', linewidth = 3,  label = 'Analytic solution (was not part of the training)')
# show a legend on the plot
plt.legend()
plt.show()