import numpy as np
import pickle
import torch
# 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
from main6 import amper, faraday
from plot_graphs import generate_data
def calc_loss(w1,time_steps,E_train,Hx_train,Hy_train,nx,ny,dt,Z,dx,dy):
    loss1 = 0.
    loss2=0.
    loss3=0.

    for n in range(time_steps):
        E,Hx,Hy = forward_function(w1, E_train[n].clone(), Hx_train[n].clone(), Hy_train[n].clone(), nx, ny, dt, Z, dx, dy, frog=2)
        loss1 +=norm2(E, E_train[n + 1])+norm2(Hx[1:nx - 1, 0:ny -1] , Hx_train[n + 1][1:nx - 1, 0:ny -1] )+norm2(Hy[0:nx - 1, 1:ny - 1], Hy_train[n + 1][0:nx - 1, 1:ny - 1])
        loss2 += (torch.square(((
                ((Hx[2:nx-1, 1:ny - 2] - Hx[1:nx - 2, 1:ny - 2]) + (Hy[1:nx - 2, 2:ny-1] - Hy[1:nx - 2, 1:ny - 2])) / (
            dx))))).mean()
        E1,Hx1,Hy1=forward_function(w1, E.clone(), Hx.clone(), Hy.clone(), nx, ny, dt, Z, dx, dy, frog=2)
        if n<time_steps-2:
           loss3 += norm2(E1, E_train[n + 2]) + norm2(Hx1[1:nx - 1, 0:ny - 1], Hx_train[n + 2][1:nx - 1, 0:ny - 1]) + norm2(
               Hy1[0:nx - 1, 1:ny - 1], Hy_train[n + 2][0:nx - 1, 1:ny - 1])

    return torch.sqrt(dt*loss1)+torch.sqrt(dt*loss2)+torch.sqrt(dt*loss3)
def norm2(A,B):
    return torch.square(A - B).mean()
def forward_function(w1, E, Hx, Hy, nx, ny, dt, Z, dx, dy, frog=2):
    filters = torch.cat(((w1 - 1) / 3, -w1, w1, (1 - w1) / 3), 0).reshape(1, 1, 4)
    bc_filters = torch.tensor([-1., 1.], dtype=float).reshape(1, 1, 2)
    E0 = E.clone()
    E01 = E.clone()
    Hx0 = Hx.clone()
    Hy0 = Hy.clone()
    Hx01 = Hx.clone()
    Hy01 = Hy.clone()
    E[1:nx - 1, 1:ny - 1] = amper(E0, Hx0, Hy0, Z, dt, dx, dy, nx, ny, bc_filters, 1)
    # E[1:nx - 1, 1:ny - 1] = amper(E0, Hx0, Hy0, Z, dt, dx, dy, nx, ny, bc_filters, 1)
    E[frog:nx - frog, frog:ny - frog] = amper(E01, Hx01, Hy01, Z, dt, dx, dy, nx, ny, filters, frog)
    Em = E.clone()
    Em1 = E.clone()

    Hx[1:nx - 1, 0:ny - 1] = faraday(Em, Hx0, Hy0, Z, dt, dx, dy, nx, ny, bc_filters, 1)[0]
    Hy[0:nx - 1, 1:ny - 1] = faraday(Em, Hx0, Hy0, Z, dt, dx, dy, nx, ny, bc_filters, 1)[1]
    Hx[frog:nx - frog, frog - 1:ny - frog] = faraday(Em1, Hx01, Hy01, Z, dt, dx, dy, nx, ny, filters, frog)[0]
    Hy[frog - 1:nx - frog, frog:ny - frog] = faraday(Em1, Hx01, Hy01, Z, dt, dx, dy, nx, ny, filters, frog)[1]

    return [E, Hx, Hy]


def create_train(nx, ny, dx, dy, dt, time_steps, k1,k2):
    E = []
    Hx = []
    Hy = []
    E_a, Hx_a, Hy_a = generate_data(1., nx, ny, k1, k2, dx=dx, dy=dy, dt=dt,
                                            time_steps=time_steps)
    E.append(E_a.copy())
    Hx.append( Hx_a.copy())
    Hy.append( Hy_a.copy())
    return [E_a, Hx_a, Hy_a]
xmin, xmax = 0.0, 1.0  # limits in the x direction
ymin, ymax = 0.0, 1.0
nx=120
ny=120
T=0.1
Z=1
time_steps=400
dt = T / time_steps  # limits in the y direction
lx = xmax - xmin  # domain length in the x direction
ly = ymax - ymin  # domain length in the y direction
dx = lx / (nx - 1)  # grid spacing in the x direction
dy = ly / (ny - 1)  # grid spacing in the y direction

w_yee=torch.tensor([1.],dtype=float,requires_grad=False)
#Loss=[]

k_train=[1.,2.]
w1=torch.tensor([1.],dtype=float,requires_grad=True)
w4=torch.tensor([27/24],dtype=float,requires_grad=False)
optimizer = torch.optim.Adam([{'params': w1},], lr=1e-2)
E_train=[]
Hx_train=[]
Hy_train=[]

for k1 in k_train:
   for k2 in k_train:
      E_a,Hx_a,Hy_a=create_train(nx, ny, dx, dy, dt, time_steps, k1,k2)
      E_train.append(E_a)
      Hx_train.append(Hx_a)
      Hy_train.append(Hy_a)

epochs=7
for i in range(epochs):
    loss=0.
    #print(w1)
    for k in range(len(E_train)):
            #print(calc_loss(w1,time_steps,E,Hx,Hy,nx,ny,dt,Z,dx,dy))
            loss += calc_loss(w1,time_steps,E_train[k].copy(),Hx_train[k].copy(),Hy_train[k].copy(),nx,ny,dt,Z,dx,dy)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('loss='+str(loss*10000))
    print(w1.grad)

with open('w1.pkl', 'rb') as file:
    # Call load method to deserialze
    w1 = pickle.load(file).detach().clone()
# k_test=k_train
# E_test = []
# Hx_test = []
# Hy_test = []
# for k1 in k_train:
#    for k2 in k_train:
#       E_a,Hx_a,Hy_a=create_train(nx, ny, dx, dy, dt, time_steps, k1,k2)
#       E_test.append(E_a)
#       Hx_test.append(Hx_a)
#       Hy_test.append(Hy_a)
#
# loss1 = 0.
# loss2 = 0.
# loss3 = 0.
#
# for k in range(len(E_test)):
#     E = E_test[k][0]
#     Hx = Hx_test[k][0]
#     Hy = Hy_test[k][0]
#
#     for n in range(time_steps):
#         w=w1
#         E,Hx,Hy = forward_function(w, E.clone(), Hx.clone(), Hy.clone(), nx, ny, dt, Z, dx, dy, frog=2)
#         loss1 +=norm2(E, E_test[k][n + 1])+norm2(Hx[1:nx - 1, 0:ny -1] , Hx_test[k][n + 1][1:nx - 1, 0:ny -1] )+norm2(Hy[0:nx - 1, 1:ny - 1], Hy_test[k][n + 1][0:nx - 1, 1:ny - 1])
#         loss2 += (torch.square(((
#                 ((Hx[1:nx, 0:ny - 1] - Hx[0:nx - 1, 0:ny - 1]) + (Hy[0:nx - 1, 1:ny] - Hy[0:nx - 1, 0:ny - 1])) / (
#             dx))))).mean()
#         E1, Hx1, Hy1 = forward_function(w, E.clone(), Hx.clone(), Hy.clone(), nx, ny, dt, Z, dx, dy, frog=2)
#
#         #if n < (time_steps - 2):
#          #  loss3 += norm2(E1, E_test[n + 2]) + norm2(Hx1[1:nx - 1, 0:ny - 1],
#            #                                        Hx_test[n + 2][1:nx - 1, 0:ny - 1]) + norm2(
#           #     Hy1[0:nx - 1, 1:ny - 1], Hy_test[n + 2][0:nx - 1, 1:ny - 1])
#         #print(torch.sqrt(loss1*dt))
# print((torch.sqrt(dt*loss2)))
# print((torch.sqrt(dt*loss1)))