import numpy as np
import pickle
import torch
from plot_graphs import forward_function,create_train,norm2
import matplotlib.pyplot as plt
k_test=[6.,7.,8.,9.,10.]
ymin, ymax = 0.0, 1.0
ny=40
xmin, xmax = 0.0, 1.0  # limits in the x direction
nx=40
T=1
Z=1
time_steps=400
dt = T / time_steps  # limits in the y direction
lx = xmax - xmin  # domain length in the x direction
ly = ymax - ymin  # domain length in the y direction
dx = lx / (nx - 1)  # grid spacing in the x direction
dy = ly / (ny - 1)  # grid spacing in the y direction

w_yee=torch.tensor([1.],dtype=float,requires_grad=False)
w4=torch.tensor([27/24],dtype=float,requires_grad=False)
with open('w1.pkl', 'rb') as file:
    # Call load method to deserialze
    w1 = pickle.load(file).detach().clone()
E_test = []
Hx_test = []
Hy_test = []
for k1 in k_test:
      E_a,Hx_a,Hy_a=create_train(nx, ny, dx, dy, dt, time_steps, k1,k1)
      E_test.append(E_a)
      Hx_test.append(Hx_a)
      Hy_test.append(Hy_a)

tot_loss=[]
tot_loss_DL = []
tot_loss_Yee = []
tot_loss4=[]
for k in range(len(k_test)):
    loss1 = 0.
    loss2 = 0.
    loss3 = 0.

    E = E_test[k][0]
    Hx = Hx_test[k][0]
    Hy = Hy_test[k][0]
    for n in range(time_steps):
        E,Hx,Hy = forward_function(w1, E.clone(), Hx.clone(), Hy.clone(), nx, ny, dt, Z, dx, dy, frog=2)
        loss1 +=norm2(E, E_test[k][n + 1])+norm2(Hx[1:nx - 1, 0:ny -1] , Hx_test[k][n + 1][1:nx - 1, 0:ny -1] )+norm2(Hy[0:nx - 1, 1:ny - 1], Hy_test[k][n + 1][0:nx - 1, 1:ny - 1])
    tot_loss_DL.append(loss1)

    loss1 = 0.
    loss2 = 0.
    loss3 = 0.
    E = E_test[k][0]
    Hx = Hx_test[k][0]
    Hy = Hy_test[k][0]
    for n in range(time_steps):
        E, Hx, Hy = forward_function(w_yee, E.clone(), Hx.clone(), Hy.clone(), nx, ny, dt, Z, dx, dy, frog=2)
        loss1 += norm2(E, E_test[k][n + 1]) + norm2(Hx[1:nx - 1, 0:ny - 1],
                                                    Hx_test[k][n + 1][1:nx - 1, 0:ny - 1]) + norm2(
            Hy[0:nx - 1, 1:ny - 1], Hy_test[k][n + 1][0:nx - 1, 1:ny - 1])
    tot_loss_Yee.append(loss1)
    loss1 = 0.
    loss2 = 0.
    loss3 = 0.
    E = E_test[k][0]
    Hx = Hx_test[k][0]
    Hy = Hy_test[k][0]
    for n in range(time_steps):
        E, Hx, Hy = forward_function(w4, E.clone(), Hx.clone(), Hy.clone(), nx, ny, dt, Z, dx, dy, frog=2)
        loss1 += norm2(E, E_test[k][n + 1]) + norm2(Hx[1:nx - 1, 0:ny - 1],
                                                    Hx_test[k][n + 1][1:nx - 1, 0:ny - 1]) + norm2(
            Hy[0:nx - 1, 1:ny - 1], Hy_test[k][n + 1][0:nx - 1, 1:ny - 1])
    tot_loss4.append(loss1)

plt.plot(k_test,tot_loss_DL,color='blue', linewidth = 3,  label = 'DL: 4-point stencil')
plt.plot(k_test,tot_loss_Yee,color='red', linewidth = 3,  label = 'Yee2 ')
plt.plot(k_test,tot_loss4,color='green', linewidth = 3,  label = 'Yee4')

plt.legend()
plt.show()
