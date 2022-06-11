# filters=torch.tensor([1.,-1.],dtype=float).reshape(1,1,2)
import numpy as np
import torch
import time
import math
import torch.nn.functional as F
from plot_graphs import generate_data
def run_p(filters,k1,k2,nx,ny,T,time_steps):
    #with torch.no_grad():
     #   filters = filters.reshape(1, 1, 2)
    xmin, xmax = 0.0, 1.0  # limits in the x direction
    ymin, ymax = 0.0, 1.0
    # dt=0.0002

    dt = T / time_steps  # limits in the y direction
    lx = xmax - xmin  # domain length in the x direction
    ly = ymax - ymin  # domain length in the y direction
    dx = lx / (nx - 1)  # grid spacing in the x direction
    dy = ly / (ny - 1)  # grid spacing in the y direction
    E_a, Hx_a, Hy_a = generate_data(xmax, nx, ny, k1, k2, dx, dy, dt, time_steps)
    E = E_a[0].clone()
    Hx = Hx_a[0].clone()
    Hy = Hy_a[0].clone()
    # E = torch.zeros((nx, ny),dtype=float)  # E_z at t=0
    # E_a = torch.zeros((nx, ny),dtype=float)
    # Hx = torch.zeros((nx, ny),dtype=float)  # at t=1/2
    # Hy = torch.zeros((nx, ny),dtype=float)  # at t=1/2

    c = math.pi * (np.sqrt(k1 ** 2 + k2 ** 2))
    P = math.pi
    # for i in range(0, nx):
    #     for j in range(0, ny):
    #         Hx[i, j] = math.sin(c * dt / 2) * (
    #                     -P * k2 * math.sin(P * k1 * dx * i) * math.cos(P * k2 * dy * (j + 1 / 2)) - P * k1 * math.sin(
    #                 P * k2 * dx * i) * math.cos(P * k1 * dy * (j + 1 / 2)))
    #         Hy[i, j] = math.sin(c * dt / 2) * (
    #                     P * k1 * math.cos(P * k1 * dx * (i+1/2)) * math.sin(P * k2 * dy * (j )) + P * k2 * math.cos(
    #                 P * k2 * dx * (i+1/2)) * math.sin(P * k1 * dy * (j )))
    #
    # for i in range(1, nx - 1):
    #     for j in range(1, ny - 1):
    #         E[i, j] = c * (math.sin(P * k1 * dx * i) * math.sin(P * k2 * dy * j) + math.sin(P * k2 * dx * i) * math.sin(
    #             P * k1 * dy * j))
    #
    # E[:, -1] = 0
    # E[-1, :] = 0

    Z = 1.

    device="cpu"



    # print(conv.weight.device)
    start_time = time.time()
    for n in range(time_steps):
        E0 = E.clone().detach()
        Hx0 = Hx.clone().detach()
        Hy0 = Hy.clone().detach()
        E[1:nx - 1, 1:ny - 1] = amper(E0,Hx0,Hy0,Z,dt,dx,dy,nx,ny,filters)
        Em=E.clone().detach()
        Hx[1:nx - 1, 0:ny - 1] = faraday(Em,Hx0,Hy0,Z,dt,dx,dy,nx,ny,filters)[0]
        Hy[0:nx - 1, 1:ny - 1] = faraday(Em,Hx0,Hy0,Z,dt,dx,dy,nx,ny,filters)[1]
        if n % 40 == 0:
            # print(E_a.dtype)
            print("{:.6f}".format((torch.square(E - E_a[n+1])).mean()))



    print("--- %s seconds ---" % (time.time() - start_time))
    # print(filters[0,0])
    #with torch.no_grad():
     #   filters = filters.reshape( 2)
      #  print(filters.requires_grad)
    return filters.sum()
def amper(E,Hnx,Hny,Z,dt,dx,dy,nx,ny,filters):
    S1 = (Z * dt / dx) * F.conv1d(torch.transpose(Hny, 0, 1).reshape(ny, 1, nx),filters).reshape(nx, ny - 1).transpose(1, 0)[
                         0:nx - 2, 1:ny - 1]
    S2 = (Z * dt / dy) * (F.conv1d(Hnx.reshape(nx, 1, ny),filters).reshape(nx, ny - 1))[1:nx - 1, 0:ny - 2]
    return E[1:nx - 1, 1:ny - 1] + S1 - S2
def faraday(E,Hnx,Hny,Z,dt,dx,dy,nx,ny,filters):
    S3 = (dt / (Z * dy)) * F.conv1d(E.reshape(nx, 1, ny),filters).reshape(nx, ny - 1)[1:nx - 1, 0:ny - 1]
    S4 = (dt / (Z * dx)) * F.conv1d(torch.transpose(E, 0, 1).reshape(ny, 1, nx),filters).reshape(ny, nx - 1).transpose(1, 0)[0:,
                           1:ny - 1]

    Ax= Hnx[1:nx - 1, 0:ny - 1] - S3
    Ay= Hny[0:nx - 1, 1:ny - 1] + S4
    return [Ax,Ay]

