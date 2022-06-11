import numpy as np
import torch
import time
import math
import torch.nn.functional as F
def MSE_pred(Z,dt,dx,dy,nx,ny,filters,frog,E,Hx,Hy,bc_filters,time_steps):
    AE=[]
    AHx=[]
    AHy=[]
    for n in range(time_steps):
        AE.append(E)
        AHx.append(Hx)
        AHy.append(Hy)
        E0 = E.detach().clone()
        E01 = E.clone()
        Hx0 = Hx.detach().clone()
        Hy0 = Hy.detach().clone()
        Hx01 = Hx.clone()
        Hy01 = Hy.clone()
        E[1:nx - 1, 1:ny - 1] = amper(E0, Hx0, Hy0, Z, dt, dx, dy, nx, ny, bc_filters, 1)
        # E[1:nx - 1, 1:ny - 1] = amper(E0, Hx0, Hy0, Z, dt, dx, dy, nx, ny, bc_filters, 1)
        # print(E.requires_grad)
        # E[1:nx - 1, 1:ny - 1] = amper(E0, Hx0, Hy0, Z, dt, dx, dy, nx, ny, bc_filters, 1)
        E[frog:nx - frog, frog:ny - frog] = amper(E01, Hx01, Hy01, Z, dt, dx, dy, nx, ny, filters, frog)
        Em = E.detach().clone()
        Em1 = E.clone()

        Hx[1:nx - 1, 0:ny - 1] = faraday(Em, Hx0, Hy0, Z, dt, dx, dy, nx, ny, bc_filters, 1)[0]
        Hy[0:nx - 1, 1:ny - 1] = faraday(Em, Hx0, Hy0, Z, dt, dx, dy, nx, ny, bc_filters, 1)[1]
        Hx[frog:nx - frog, frog - 1:ny - frog] = faraday(Em1, Hx01, Hy01, Z, dt, dx, dy, nx, ny, filters, frog)[0]
        Hy[frog - 1:nx - frog, frog:ny - frog] = faraday(Em1, Hx01, Hy01, Z, dt, dx, dy, nx, ny, filters, frog)[1]


    return [AE,AHx,AHy]


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
