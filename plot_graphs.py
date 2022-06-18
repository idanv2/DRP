import numpy as np
import math
import torch
from torch.functional import F
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

def generate_data(xmax, nx,ny,k1,k2,dx,dy,dt,time_steps):
    c = math.pi * (np.sqrt(k1 ** 2 + k2 ** 2))
    P = math.pi
    x=torch.linspace(0.,xmax,nx)
    X,Y =torch.meshgrid(x, x, indexing='ij')
    E=[]
    Hx=[]
    Hy=[]
    for n in range(time_steps+1):
        E.append(( c * np.cos(c * n * dt) * (torch.sin(P * k1 * X) * torch.sin(P * k2 * Y) + torch.sin(P * k2 * X) * torch.sin(
            P * k1 * Y))).type(torch.float64))
        Hx.append(( np.sin(c * (dt / 2) * (2 * n + 1)) * (
                -P * k2 * torch.sin(P * k1 * X) * torch.cos(P * k2 * (Y + dy / 2)) - P * k1 * torch.sin(
            P * k2 * X) * torch.cos(P * k1 * (Y + dy / 2)))).type(torch.float64))
        Hy.append(( np.sin(c * (dt / 2) * (2 * n + 1)) * (
                P * k1 * torch.cos(P * k1 * (X + dx / 2)) * torch.sin(P * k2 * Y) + P * k2 * torch.cos(
            P * k2 * (X + dx / 2)) * torch.sin(P * k1 * Y))).type(torch.float64))

    return  [E,Hx,Hy]
#E_a,Hx_a,Hy_a=generate_data()
def calc_loss(w1,time_steps,E_train,Hx_train,Hy_train,nx,ny,dt,Z,dx,dy):
    loss1 = 0.
    loss2=0.
    loss3=0.

    for n in range(time_steps):
        E,Hx,Hy = forward_function(w1, E_train[n].clone(), Hx_train[n].clone(), Hy_train[n].clone(), nx, ny, dt, Z, dx, dy, frog=2)
        loss1 +=norm2(E, E_train[n + 1])+norm2(Hx[1:nx - 1, 0:ny -1] , Hx_train[n + 1][1:nx - 1, 0:ny -1] )+norm2(Hy[0:nx - 1, 1:ny - 1], Hy_train[n + 1][0:nx - 1, 1:ny - 1])
        loss2 += (torch.square(((
                ((Hx[1:nx, 0:ny - 1] - Hx[0:nx - 1, 0:ny - 1]) + (Hy[0:nx - 1, 1:ny] - Hy[0:nx - 1, 0:ny - 1])) / (
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
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]