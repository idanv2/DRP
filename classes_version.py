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
    frog=2;batch_size=10
    nx=20;ny=20;ymin, ymax = 0.0, 1.0;xmin, xmax = 0.0, 1.0;Z=1
    T=1;time_steps = 400; dt = T / time_steps
    lx = xmax - xmin;ly = ymax - ymin;dx = lx / (nx - 1);dy = ly / (ny - 1)
    w_yee = torch.tensor([1.],dtype=float)

class Network(par,nn.Module):

    def __init__(self,w1):
        #super(Network,self).__init__()
        self.params = []
        self.params.append(w1)
        self.optimizer=torch.optim.Adam([{'params': w1},], lr=1e-2)

    def forward(self,X1,X2,X3):
         E=X1.clone()
         Hx=X2.clone()
         Hy=X3.clone()
         w1=self.params[0]
         filters = torch.cat(((w1 - 1) / 3, -w1, w1, (1 - w1) / 3), 0).reshape(1, 1, 4)
         bc_filters = torch.tensor([-1., 1.], dtype=float).reshape(1, 1, 2)
         E0 = E.clone()
         E01 = E.clone()
         Hx0 = Hx.clone()
         Hy0 = Hy.clone()
         Hx01 = Hx.clone()
         Hy01 = Hy.clone()
         E[1:par.nx - 1, 1:par.ny - 1] = amper(E0, Hx0, Hy0, par.Z, par.dt, par.dx, par.dy, par.nx, par.ny, bc_filters, 1)
         # E[1:nx - 1, 1:ny - 1] = amper(E0, Hx0, Hy0, Z, dt, dx, dy, nx, ny, bc_filters, 1)
         E[par.frog:par.nx - par.frog, par.frog:par.ny - par.frog] = amper(E01, Hx01, Hy01, par.Z, par.dt, par.dx, par.dy, par.nx, par.ny, filters, par.frog)
         Em = E.clone()
         Em1 = E.clone()

         Hx[1:par.nx - 1, 0:par.ny - 1] = faraday(Em, Hx0, Hy0, par.Z, par.dt, par.dx, par.dy, par.nx, par.ny, bc_filters, 1)[0]
         Hy[0:par.nx - 1, 1:par.ny - 1] = faraday(Em, Hx0, Hy0, par.Z, par.dt, par.dx, par.dy, par.nx, par.ny, bc_filters, 1)[1]
         Hx[par.frog:par.nx - par.frog, par.frog - 1:par.ny - par.frog] = faraday(Em1, Hx01, Hy01, par.Z, par.dt, par.dx, par.dy, par.nx, par.ny, filters, par.frog)[0]
         Hy[par.frog - 1:par.nx - par.frog, par.frog:par.ny - par.frog] = faraday(Em1, Hx01, Hy01, par.Z, par.dt, par.dx, par.dy, par.nx, par.ny, filters, par.frog)[1]

         return [E, Hx, Hy]
    def loss(self,x,y):
        q=0
        for k in range(len(x)):
            q+=norm2(x[k],y[k])
        return q



N=Network(torch.tensor([1.],dtype=float,requires_grad=True))
k1=1.
k2=1.
E_a,Hx_a,Hy_a=create_train(par.nx, par.ny, par.dx, par.dy, par.dt, par.time_steps,k1,k1)
data=[]
for k in range(par.time_steps):
    x=torch.cat((E_a[k],Hx_a[k],Hy_a[k]),0)
    y=torch.cat((E_a[k+1],Hx_a[k+1],Hy_a[k+1]),0)
    data.append((x,y))
train_loader = torch.utils.data.DataLoader(dataset=Custom_Dataset(data),
                                           batch_size=par.batch_size,
                                           shuffle=False)
# training loop
loss=0
for i,data in enumerate(train_loader):

    for k in range(par.batch_size):
        f=N.forward(data[0][k,0:par.nx,0:par.ny],data[0][k,par.nx:2*par.nx,0:par.nx],data[0][k,2*par.nx:3*par.nx,0:par.nx])
        loss+=N.loss(f,[data[1][k,0:par.nx,0:par.ny],data[1][k,par.nx:2*par.nx,0:par.nx],data[1][k,2*par.nx:3*par.nx,0:par.nx]])
        print(loss)




#print(v[-1,:,:]-x)


#N=Network(par.w_yee)
#print(type(Train[0]))
#print(Train[0])
#print(N.params)
#N.forward(E_a[1],Hx_a[1],Hy_a[1])



