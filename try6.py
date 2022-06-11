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

loss=0
#k1=1.
#k2=1.
#E_a, Hx_a, Hy_a = generate_data(1., nx, ny, k1 ,k2, dx=dx, dy=dy, dt=dt,
#                                            time_steps=time_steps)

#print(run_p(w1, k1, k2, nx, ny, T, time_steps, E_a=E_a, Hx_a=Hx_a, Hy_a=Hy_a, dt=dt,
#                         dx=dx, dy=dy))
w1=torch.tensor([27/24],dtype=float,requires_grad=True)
w_yee=torch.tensor([1.],dtype=float,requires_grad=False)
w0=w1.detach().clone()
L=[1.,3.]
L_validate=[2.,4.]
Loss=[]
Loss_validate=[]
epochs=6
optimizer = torch.optim.Adam([
    {'params': w1},
], lr=1e-2)
for i in range(epochs):
    loss=0

    for k1 in L:
        for k2 in L:
            E_a, Hx_a, Hy_a = generate_data(1., nx, ny, k1, k2, dx=dx, dy=dy, dt=dt,
                                            time_steps=time_steps)
            loss = loss+run_p(w1, k1, k2, nx, ny, T, time_steps, E_a=E_a, Hx_a=Hx_a, Hy_a=Hy_a, dt=dt,
                         dx=dx, dy=dy,training=True)

    optimizer.zero_grad()
    #print("{:.7f}".format(loss))
    loss.backward()

    Loss.append(loss.detach().clone())
    loss_val = 0
    for k1 in L_validate:
        for k2 in L_validate:
            E_a, Hx_a, Hy_a = generate_data(1., nx, ny, k1, k2, dx=dx, dy=dy, dt=dt,
                                            time_steps=time_steps)
            loss_val = loss_val+run_p(w1.detach().clone(), k1, k2, nx, ny, T, time_steps, E_a=E_a, Hx_a=Hx_a, Hy_a=Hy_a, dt=dt,
                         dx=dx, dy=dy,training=True)
    optimizer.step()
    print('loss='+str(loss))
    Loss_validate.append(loss_val.detach().clone())
plt.plot(range(epochs),torch.stack(Loss)-torch.stack(Loss).mean())
plt.plot(range(epochs),torch.stack(Loss_validate)-torch.stack(Loss_validate).mean())
#print(torch.stack(Loss)-torch.stack(Loss).mean())
plt.show()
#print((torch.sum(Loss.detach())))
w_new=w1.clone()
# Open a file and use dump()
with open('w1.pkl', 'wb') as file:
    # A new file will be created
    pickle.dump(w1, file)

