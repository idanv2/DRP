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
        example, target1,target2 = self.dataset[index]
        return np.array(example), target1,target2

    def __len__(self):
        return len(self.dataset)
class par:
    lr = 1e-2;batch_size = 10;
    frog=2; frog1=2
    nx=40;ny=40;ymin, ymax = 0.0, 1.0;xmin, xmax = 0.0, 1.0;Z=1
    T=1;time_steps = 400; dt = T / time_steps
    lx = xmax - xmin;ly = ymax - ymin;dx = lx / (nx - 1);dy = ly / (ny - 1)
    w_yee = torch.tensor([1.],dtype=float)
    bc_filters = torch.tensor([-1., 1.], dtype=float).reshape(1, 1, 2)


class Network(par,nn.Module):

    def __init__(self,w1):
        #super(Network,self).__init__()
        self.params = []
        self.params.append(w1)
        self.optimizer=torch.optim.Adam([{'params': self.params},], lr=par.lr)

    def forward(self,X1,X2,X3):
         E=X1.clone()
         Hx=X2.clone()
         Hy=X3.clone()
         w1=self.params[0]
         filters = torch.cat(((w1 - 1) / 3, -w1, w1, (1 - w1) / 3), 0).reshape(1, 1, 4)
         E[1:par.nx - 1, 1:par.ny - 1] = self.amper(X1.clone(), X2.clone(), X3.clone(),par.bc_filters,1)
         E[par.frog:par.nx - par.frog, par.frog:par.ny - par.frog] = self.amper(X1.clone(), X2.clone(), X3.clone(),filters,par.frog)
         Hx[1:par.nx - 1, 0:par.ny - 1] = self.faraday(E.clone(), X2.clone(), X3.clone(),par.bc_filters,1)[0]
         Hy[0:par.nx - 1, 1:par.ny - 1] = self.faraday(E.clone(), X2.clone(), X3.clone(),par.bc_filters,1)[1]
         Hx[par.frog:par.nx - par.frog, par.frog - 1:par.ny - par.frog] = self.faraday(E.clone(), X2.clone(), X3.clone(), filters,par.frog)[0]
         Hy[par.frog - 1:par.nx - par.frog, par.frog:par.ny - par.frog] = self.faraday(E.clone(), X2.clone(), X3.clone(), filters,par.frog)[1]
         return [E, Hx, Hy]
    def amper(self,E,Hx,Hy,filters,step):
        S1 = (par.Z * par.dt / par.dx) * F.conv1d(torch.transpose(Hy, 0, 1).reshape(par.ny, 1, par.nx), filters).reshape(par.ny, par.nx - (
                    2 * step - 1)).transpose(1, 0)[
                             0:-1, step:par.ny - step]
        S2 = (par.Z * par.dt / par.dy) * (F.conv1d(Hx.reshape(par.nx, 1, par.ny), filters).reshape(par.nx, par.ny - (2 * step - 1)))[
                             step:par.nx - step, 0:-1]
        return E[step:par.nx - step, step:par.ny - step] + S1 - S2
    def faraday(self,En,Hnx,Hny,filters,step):
        S3 = (par.dt / (par.Z * par.dy)) * F.conv1d(En.reshape(par.nx, 1, par.ny), filters).reshape(par.nx, par.ny - (2 * step - 1))[step:par.nx - step,
                               0:]
        # print('hhh')
        S4 = (par.dt / (par.Z * par.dx)) * F.conv1d(torch.transpose(En, 0, 1).reshape(par.ny, 1, par.nx), filters).reshape(par.ny, par.nx - (
                    2 * step - 1)).transpose(1, 0)[0:,
                               step:par.ny - step]

        Ax = Hnx[step:par.nx - step, step - 1:par.ny - step] - S3
        Ay = Hny[step - 1:par.nx - step, step:par.ny - step] + S4
        return [Ax, Ay]
    def loss(self,E_train,Hx_train,Hy_train):
            n=0
            E, Hx, Hy = self.forward(E_train[n],Hx_train[n],Hy_train[n])
            loss1 = norm2(E, E_train[n + 1]) + norm2(Hx[1:par.nx - 1, 0:par.ny - 1],
                                                      Hx_train[n + 1][1:par.nx - 1, 0:par.ny - 1]) + norm2(
                Hy[0:par.nx - 1, 1:par.ny - 1], Hy_train[n + 1][0:par.nx - 1, 1:par.ny - 1])
            loss2 = (torch.square(((
                    ((Hx[2:par.nx - 1, 1:par.ny - 2] - Hx[1:par.nx - 2, 1:par.ny - 2]) + (
                                Hy[1:par.nx - 2, 2:par.ny - 1] - Hy[1:par.nx - 2, 1:par.ny - 2])) / (
                        par.dx))))).mean()
            E1, Hx1, Hy1 = self.forward( E.clone(), Hx.clone(), Hy.clone())

            loss3 = norm2(E1, E_train[n + 2]) + norm2(Hx1[1:par.nx - 1, 0:par.ny - 1],
                                                           Hx_train[n + 2][1:par.nx - 1, 0:par.ny - 1]) + norm2(
                    Hy1[0:par.nx - 1, 1:par.ny - 1], Hy_train[n + 2][0:par.nx - 1, 1:par.ny - 1])

            return torch.sqrt(loss1),torch.sqrt(loss2),torch.sqrt(loss3)






N=Network(torch.tensor([1.],dtype=float,requires_grad=True))

k_train=torch.tensor([1.])
w1=torch.tensor([1.],dtype=float,requires_grad=True)
w4=torch.tensor([27/24],dtype=float,requires_grad=False)
E_train=[]
Hx_train=[]
Hy_train=[]

for k1 in k_train:
   for k2 in k_train:
      E_a,Hx_a,Hy_a=create_train(par.nx, par.ny, par.dx, par.dy, par.dt, par.time_steps, k1,k2)
      E_train.append(E_a.copy())
      Hx_train.append(Hx_a.copy())
      Hy_train.append(Hy_a.copy())








#
data=[]
for k in range(len(E_train)):
    for n in range(par.time_steps):
        if n < par.time_steps - 2:
            x = torch.cat((E_a[n], Hx_a[n], Hy_a[n]), 0)
            y = torch.cat((E_a[n + 1], Hx_a[n + 1], Hy_a[n + 1]), 0)
            z = torch.cat((E_a[n + 2], Hx_a[n + 2], Hy_a[n + 2]), 0)
        else:
            x = torch.cat((E_a[n], Hx_a[n], Hy_a[n]), 0)
            y = torch.cat((E_a[n + 1], Hx_a[n + 1], Hy_a[n + 1]), 0)
            z = torch.cat((E_a[n] * 0, Hx_a[n] * 0, Hy_a[n] * 0), 0)

        data.append((x, y, z))
train_loader = torch.utils.data.DataLoader(dataset=Custom_Dataset(data),
                                            batch_size=par.batch_size,
                                            shuffle=False)
# # # training loop
loss=0
#i*batches=samples

epochs=7
for i in range(epochs):
    print(N.params)

    for i, data in enumerate(train_loader):
        loss = 0.
        for k in range(par.batch_size):
            E = (data[0][k, 0:par.nx, 0:par.ny].clone(), data[1][k, 0:par.nx, 0:par.ny].clone(), data[2][k, 0:par.nx, 0:par.ny].clone())
            Hx = (data[0][k, par.nx:2 * par.nx, 0:par.nx].clone(), data[1][k, par.nx:2 * par.nx, 0:par.nx].clone(),
                  data[2][k, par.nx:2 * par.nx, 0:par.nx].clone())
            Hy = (data[0][k, 2 * par.nx:3 * par.nx, 0:par.nx].clone(), data[1][k, 2 * par.nx:3 * par.nx, 0:par.nx].clone(),
                  data[2][k, 2 * par.nx:3 * par.nx, 0:par.nx].clone())
            loss1, loss2, loss3 = N.loss(E, Hx, Hy)
            loss+=(loss1+loss2+loss3)
        N.optimizer.zero_grad()
        loss.backward()
        N.optimizer.step()
    print('loss='+str(loss))
    print(N.params)