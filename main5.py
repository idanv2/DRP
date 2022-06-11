# filters=torch.tensor([1.,-1.],dtype=float).reshape(1,1,2)
import numpy as np
import torch
import time
import math
import torch.nn.functional as F
from plot_graphs import generate_data
import tensorflow as tf
def run_p(w1,k1,k2,nx,ny,T,time_steps,E_a, Hx_a, Hy_a,dt,dx,dy):
    frog=2

    #xmin, xmax = 0.0, 1.0  # limits in the x direction
    # dt=0.0002

   # dt = T / time_steps  # limits in the y direction
    #lx = xmax - xmin  # domain length in the x direction
    #ly = ymax - ymin  # domain length in the y direction
    #dx = lx / (nx - 1)  # grid spacing in the x direction
    #dy = ly / (ny - 1)  # grid spacing in the y direction
    filters = torch.cat(((w1 - 1) / 3, -w1 , w1, (1 - w1) / 3), 0).reshape(1, 1, 4)
    #print(filters.requires_grad)

    #E_a, Hx_a, Hy_a = generate_data(xmax, nx, ny, k1, k2, dx, dy, dt, time_steps)
    E = E_a[0].clone()
    Hx = Hx_a[0].clone()
    Hy = Hy_a[0].clone()

    c = math.pi * (np.sqrt(k1 ** 2 + k2 ** 2))
    P = math.pi
    Z = 1.
    device="cpu"
    start_time = time.time()
    bc_filters = torch.tensor( [-1., 1.], dtype=float).reshape(1, 1, 2)
    loss=0
    #ymin, ymax = 0.0, 1.0
    for n in range(time_steps):
        E0 = E_a[n].detach().clone()
        E01 = E_a[n].clone()
        Hx0 = Hx_a[n].detach().clone()
        Hy0 = Hy_a[n].detach().clone()
        Hx01 = Hx_a[n].clone()
        Hy01 = Hy_a[n].clone()
        E[1:nx-1,1:ny-1]=amper(E0,Hx0,Hy0,Z,dt,dx,dy,nx,ny,bc_filters,1)
        #print(E.requires_grad)
        #E[1:nx - 1, 1:ny - 1] = amper(E0, Hx0, Hy0, Z, dt, dx, dy, nx, ny, bc_filters, 1)
        E[frog:nx - frog, frog:ny - frog] = amper(E01,Hx01,Hy01,Z,dt,dx,dy,nx,ny,filters,frog)
        Em = E_a[n + 1].detach().clone()
        Em1 = E_a[n + 1].clone()
        Hx[1:nx - 1, 0:ny - 1] = faraday(Em, Hx0, Hy0, Z, dt, dx, dy, nx, ny, bc_filters, 1)[0]
        Hy[0:nx - 1, 1:ny - 1] = faraday(Em, Hx0, Hy0, Z, dt, dx, dy, nx, ny, bc_filters, 1)[1]
        Hx[frog:nx - frog, frog - 1:ny - frog] = faraday(Em1, Hx01, Hy01, Z, dt, dx, dy, nx, ny, filters, frog)[0]
        Hy[frog - 1:nx - frog, frog:ny - frog] = faraday(Em1, Hx01, Hy01, Z, dt, dx, dy, nx, ny, filters, frog)[1]



            # print(E_a.dtype)
        frog2=1
        #print("{:.6f}".format((torch.square(E[frog2:nx - frog2, frog2:ny - frog2] - E_a[n+1][frog2:nx - frog2, frog2:ny - frog2])).mean()))
        loss = loss + (torch.square(E[frog2:nx - frog2, frog2:ny - frog2] - E_a[n+1][frog2:nx - frog2, frog2:ny - frog2])).mean()




    return torch.sqrt(loss)
def amper(E,Hnx,Hny,Z,dt,dx,dy,nx,ny,filters,frog):
    S1 = (Z * dt / dx) * F.conv1d(torch.transpose(Hny, 0, 1).reshape(ny, 1, nx),filters).reshape(ny, nx - (2*frog-1)).transpose(1, 0)[
                         0:-1, frog:ny - frog]
    S2 = (Z * dt / dy) * (F.conv1d(Hnx.reshape(nx, 1, ny),filters).reshape(nx, ny-(2*frog-1)))[frog:nx - frog, 0:-1]
    return E[frog:nx - frog, frog:ny - frog] + S1 - S2


def faraday(E,Hnx,Hny,Z,dt,dx,dy,nx,ny,filters,frog):
    S3 = (dt / (Z * dy)) * F.conv1d(E.reshape(nx, 1, ny),filters).reshape(nx, ny - (2*frog-1))[frog:nx - frog, 0:]
    #print('hhh')
    S4 = (dt / (Z * dx)) * F.conv1d(torch.transpose(E, 0, 1).reshape(ny, 1, nx),filters).reshape(ny, nx -(2*frog-1)).transpose(1, 0)[0:,
                           frog:ny - frog]

    Ax= Hnx[frog:nx - frog, frog-1:ny - frog] - S3
    Ay= Hny[frog-1:nx - frog, frog:ny - frog] + S4
    return [Ax,Ay]

