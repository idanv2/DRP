import tensorflow as tf
import numpy as np
import torch
import torch.nn as nn
from  torch.functional import F
import torchvision
from torch.utils.data import Dataset,DataLoader
from plot_graphs import *

class Custom_Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, _dataset):
        self.dataset = _dataset

    def __getitem__(self, index):
        example, target = self.dataset[index]
        return np.array(example), target

    def __len__(self):
        return len(self.dataset)
class par:
    frog1=2
    frog=2;batch_size=10
    nx=20;ny=20;ymin, ymax = 0.0, 1.0;xmin, xmax = 0.0, 1.0;Z=1
    T=1;time_steps = 400; dt = T / time_steps
    lx = xmax - xmin;ly = ymax - ymin;dx = lx / (nx - 1);dy = ly / (ny - 1)
    w_yee = torch.tensor([1.],dtype=float)
    bc_filters = torch.tensor([-1., 1.], dtype=float).reshape(1, 1, 2)


class Network(par,nn.Module):

    def __init__(self,w1):
        #super(Network,self).__init__()
        self.params = []
        self.params.append(w1)
        self.optimizer=torch.optim.Adam([{'params': self.params},], lr=1e-2)

    def forward(self,X1,X2,X3):
         E=X1.clone()
         Hx=X2.clone()
         Hy=X3.clone()
         w1=self.params[0]
         filters = torch.cat(((w1 - 1) / 3, -w1, w1, (1 - w1) / 3), 0).reshape(1, 1, 4)
         E[1:par.nx - 1, 1:par.ny - 1] = self.amper(X1, X2, X3,par.bc_filters,1)
         E[par.frog:par.nx - par.frog, par.frog:par.ny - par.frog] = self.amper(E.clone(), Hx.clone(), Hy.clone(),filters,par.frog)
         Hx[1:par.nx - 1, 0:par.ny - 1] = self.faraday(X1, X2, X3,par.bc_filters,1)[0]
         Hy[0:par.nx - 1, 1:par.ny - 1] = self.faraday(X1, X2, X3,par.bc_filters,1)[1]
         Hx[par.frog:par.nx - par.frog, par.frog - 1:par.ny - par.frog] = self.faraday(E.clone(), Hx.clone(), Hy.clone(), filters,par.frog)[0]
         Hy[par.frog - 1:par.nx - par.frog, par.frog:par.ny - par.frog] = self.faraday(E.clone(), Hx.clone(), Hy.clone(), filters,par.frog)[1]
         return [E, Hx, Hy]
    def amper(self,E,Hx,Hy,filters,step):
        S1 = (par.Z * par.dt / par.dx) * F.conv1d(torch.transpose(Hy, 0, 1).reshape(par.ny, 1, par.nx), filters).reshape(par.ny, par.nx - (
                    2 * step - 1)).transpose(1, 0)[
                             0:-1, step:par.ny - step]
        S2 = (par.Z * par.dt / par.dy) * (F.conv1d(Hx.reshape(par.nx, 1, par.ny), filters).reshape(par.nx, par.ny - (2 * step - 1)))[
                             step:par.nx - step, 0:-1]
        return E[step:par.nx - step, step:par.ny - step] + S1 - S2
    def faraday(self,E,Hx,Hy,filters,step):
        S3 = (par.dt / (par.Z * par.dy)) * F.conv1d(E.reshape(par.nx, 1, par.ny), filters).reshape(par.nx, par.ny - (2 * step - 1))[step:par.nx - step,
                               0:]
        S4 = (par.dt / (par.Z * par.dx)) * F.conv1d(torch.transpose(E, 0, 1).reshape(par.ny, 1, par.nx), filters).reshape(par.ny, par.nx - (
                    2 * step - 1)).transpose(1, 0)[0:,
                               step:par.ny - step]

        Ax = Hx[step:par.nx - step, step - 1:par.ny - step] - S3
        Ay = Hy[step - 1:par.nx - step, step:par.ny - step] + S4
        return [Ax, Ay]
    def loss(self,E_train,Hx_train,Hy_train):
        loss1=0
        loss2=0
        loss3=0
        for n in range(par.time_steps):
            E, Hx, Hy = self.forward(E_train[n],Hx_train[n],Hy_train[n])
            loss1 += norm2(E, E_train[n + 1]) + norm2(Hx[1:par.nx - 1, 0:par.ny - 1],
                                                      Hx_train[n + 1][1:par.nx - 1, 0:par.ny - 1]) + norm2(
                Hy[0:par.nx - 1, 1:par.ny - 1], Hy_train[n + 1][0:par.nx - 1, 1:par.ny - 1])
            loss2 += (torch.square(((
                    ((Hx[2:par.nx - 1, 1:par.ny - 2] - Hx[1:par.nx - 2, 1:par.ny - 2]) + (
                                Hy[1:par.nx - 2, 2:par.ny - 1] - Hy[1:par.nx - 2, 1:par.ny - 2])) / (
                        par.dx))))).mean()
            E1, Hx1, Hy1 = self.forward( E.clone(), Hx.clone(), Hy.clone())
            if n < par.time_steps - 2:
                loss3 += norm2(E1, E_train[n + 2]) + norm2(Hx1[1:par.nx - 1, 0:par.ny - 1],
                                                           Hx_train[n + 2][1:par.nx - 1, 0:par.ny - 1]) + norm2(
                    Hy1[0:par.nx - 1, 1:par.ny - 1], Hy_train[n + 2][0:par.nx - 1, 1:par.ny - 1])

        return torch.sqrt(par.dt*loss1)+torch.sqrt(par.dt*loss2)+torch.sqrt(par.dt*loss3)






N=Network(torch.tensor([1.],dtype=float,requires_grad=True))
k_train=[1.,2.]
w1=torch.tensor([1.],dtype=float,requires_grad=True)
w4=torch.tensor([27/24],dtype=float,requires_grad=False)
optimizer = N.optimizer
E_train=[]
Hx_train=[]
Hy_train=[]

for k1 in k_train:
   for k2 in k_train:
      E_a,Hx_a,Hy_a=create_train(par.nx, par.ny, par.dx, par.dy, par.dt, par.time_steps, k1,k2)
      E_train.append(E_a)
      Hx_train.append(Hx_a)
      Hy_train.append(Hy_a)

epochs=7
for i in range(epochs):
    loss=0.
    #print(w1)
    for k in range(len(E_train)):
            #print(calc_loss(w1,time_steps,E,Hx,Hy,nx,ny,dt,Z,dx,dy))
            loss += N.loss(E_train[k].copy(),Hx_train[k].copy(),Hy_train[k].copy())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('loss='+str(loss*10000))
    print(N.params)


# k1=1.
# k2=1.
# E_a,Hx_a,Hy_a=create_train(par.nx, par.ny, par.dx, par.dy, par.dt, par.time_steps,k1,k1)
# data=[]
# E1=E_a[4]
# Hx1=Hx_a[4]
# Hy1=Hy_a[4]
# y=N.forward(E1,Hx1,Hy1)
# N.optimizer.zero_grad()
# loss=N.loss(E1,y[0])
# loss.backward()
# #print(N.params[0])
# N.optimizer.step()
# #print(N.params[0])
# y=N.forward(E1,Hx1,Hy1)
# N.optimizer.zero_grad()
# loss=N.loss(E1,y[0])
# loss.backward()
# #print(N.params[0])
# N.optimizer.step()
# #print(N.params[0])
#
# # for k in range(par.time_steps):
# #     x=torch.cat((E_a[k],Hx_a[k],Hy_a[k]),0)
# #     y=torch.cat((E_a[k+1],Hx_a[k+1],Hy_a[k+1]),0)
# #     data.append((x,y))
# # train_loader = torch.utils.data.DataLoader(dataset=Custom_Dataset(data),
# #                                            batch_size=par.batch_size,
# #                                            shuffle=False)
# # # training loop
# # loss=0
# # for i,data in enumerate(train_loader):
# #
# #     for k in range(par.batch_size):
# #         f=N.forward(data[0][k,0:par.nx,0:par.ny],data[0][k,par.nx:2*par.nx,0:par.nx],data[0][k,2*par.nx:3*par.nx,0:par.nx])
# #         loss+=N.loss(f,[data[1][k,0:par.nx,0:par.ny],data[1][k,par.nx:2*par.nx,0:par.nx],data[1][k,2*par.nx:3*par.nx,0:par.nx]])
# #         print(loss)
# #
# #
# #
#
#
#
