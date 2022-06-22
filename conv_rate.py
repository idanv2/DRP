import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
import math
import matplotlib.animation as animation
from functions_for_torch import generate_data


def run_p(w1, k1, k2, nx, ny, T, time_steps, E_a, Hx_a, Hy_a, dt, dx, dy):
    print(w1)
    frog = 2
    # print(E_a[0].requires_grad)
    filters = torch.cat(((w1 - 1) / 3, -w1, w1, (1 - w1) / 3), 0).reshape(1, 1, 4)
    E = E_a[0].clone()
    Hx = Hx_a[0].clone()
    Hy = Hy_a[0].clone()
    c = math.pi * (np.sqrt(k1 ** 2 + k2 ** 2))
    P = math.pi
    Z = 1
    bc_filters = torch.tensor([-1., 1.], dtype=float).reshape(1, 1, 2)
    tot_e = []
    tot_hx = []
    for n in range(time_steps):
        tot_e.append(E.clone())
        tot_hx.append(Hx.clone())
        E0 = E.clone()
        E01 = E.clone()
        Hx0 = Hx.clone()
        Hy0 = Hy.clone()
        Hx01 = Hx.clone()
        Hy01 = Hy.clone()
        # print(E.dtype)
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
    l = 0.
    for n in range(len(tot_e)):
        l = l + (abs(tot_e[n] - E_a[n]).max())

    l = l / (time_steps)
    return [torch.log(l).cpu().detach().numpy(), np.log(dx)]


def amper(E, Hnx, Hny, Z, dt, dx, dy, nx, ny, filters, frog):
    S1 = (Z * dt / dx) * F.conv1d(torch.transpose(Hny, 0, 1).reshape(ny, 1, nx), filters).reshape(ny, nx - (
            2 * frog - 1)).transpose(1, 0)[
                         0:-1, frog:ny - frog]
    S2 = (Z * dt / dy) * (F.conv1d(Hnx.reshape(nx, 1, ny), filters).reshape(nx, ny - (2 * frog - 1)))[
                         frog:nx - frog, 0:-1]
    return E[frog:nx - frog, frog:ny - frog] + S1 - S2


def faraday(E, Hnx, Hny, Z, dt, dx, dy, nx, ny, filters, frog):
    S3 = (dt / (Z * dy)) * F.conv1d(E.reshape(nx, 1, ny), filters).reshape(nx, ny - (2 * frog - 1))[frog:nx - frog,
                           0:]
    # print('hhh')
    S4 = (dt / (Z * dx)) * F.conv1d(torch.transpose(E, 0, 1).reshape(ny, 1, nx), filters).reshape(ny, nx - (
            2 * frog - 1)).transpose(1, 0)[0:,
                           frog:ny - frog]

    Ax = Hnx[frog:nx - frog, frog - 1:ny - frog] - S3
    Ay = Hny[frog - 1:nx - frog, frog:ny - frog] + S4
    return [Ax, Ay]



logy=[]
logx=[]
with open('w1.pkl', 'rb') as file:
    # Call load method to deserialze
    w1 = pickle.load(file).detach().clone()
w_4 = torch.tensor([27 / 24], dtype=float, requires_grad=False)
w_yee = torch.tensor([1.], dtype=float, requires_grad=False)
ln=[25,50,100,150]
for nx in ln:
    k1, k2 = 1., 1.

    xmin, xmax = 0.0, 1.0  # limits in the x direction
    ny = nx  # number of points in the y direction
    ymin, ymax = 0.0, 1.0
    # dt=0.0002
    T = 0.1
    time_steps = 600
    dt = T / time_steps
    time_steps_2 = time_steps  # limits in the y direction
    lx = xmax - xmin  # domain length in the x direction
    ly = ymax - ymin  # domain length in the y direction
    dx = lx / (nx - 1)  # grid spacing in the x direction
    dy = ly / (ny - 1)  # grid spacing in the y direction

    c = math.pi * (np.sqrt(k1 ** 2 + k2 ** 2))
    P = math.pi
    x = np.linspace(0., xmax, nx)

    E_a, Hx_a, Hy_a = generate_data(1., nx, ny, k1, k2, dx=dx, dy=dy, dt=dt, time_steps=time_steps)
    y, x = run_p(w1, k1, k2, nx, ny, T, time_steps, E_a, Hx_a, Hy_a, dt, dx, dy)
    logx.append(x.copy())
    logy.append(y.copy())
print(np.diff(logy)/np.diff(logx))