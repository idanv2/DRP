import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import torch
import math
import matplotlib.animation as animation
from experiments import generate_html
# Grid parameters.
k1, k2 = 15., 15.
nx = 80  # number of points in the x direction
xmin, xmax = 0.0, 1.0  # limits in the x direction
ny = nx  # number of points in the y direction
ymin, ymax = 0.0, 1.0
# dt=0.0002
T = 1
time_steps = 300
dt = T / time_steps
time_steps_2 = time_steps * 2  # limits in the y direction
lx = xmax - xmin  # domain length in the x direction
ly = ymax - ymin  # domain length in the y direction
dx = lx / (nx - 1)  # grid spacing in the x direction
dy = ly / (ny - 1)  # grid spacing in the y direction

c = math.pi * (np.sqrt(k1 ** 2 + k2 ** 2))
P = math.pi
x = np.linspace(0., xmax, nx)
X, Y = np.meshgrid(x, x, indexing='ij')
E_a = []
Hx_a = []
Hy_a = []
for n in range(time_steps_2 + 1):
    E_a.append((c * np.cos(c * n * dt) * (np.sin(P * k1 * X) * np.sin(P * k2 * Y) + np.sin(P * k2 * X) * np.sin(
        P * k1 * Y))))
    Hx_a.append(np.sin(c * (dt / 2) * (2 * n + 1)) * (
            -P * k2 * np.sin(P * k1 * X) * np.cos(P * k2 * (Y + dy / 2)) - P * k1 * np.sin(
        P * k2 * X) * np.cos(P * k1 * (Y + dy / 2))))
    Hy_a.append(np.sin(c * (dt / 2) * (2 * n + 1)) * (
            P * k1 * np.cos(P * k1 * (X + dx / 2)) * np.sin(P * k2 * Y) + P * k2 * np.cos(
        P * k2 * (X + dx / 2)) * np.sin(P * k1 * Y)))

E = E_a[0].copy()
Hx = Hx_a[0].copy()
Hy = Hy_a[0].copy()
Z = 1
Loss_H = []
Loss_E = []
tot_e=[]
tot_hx=[]
for n in range(time_steps_2):
    tot_e.append(E.copy())
    tot_hx.append(Hx.copy())
    loss_H = 0
    loss_E = 0
    En = E.copy()
    Hnx = Hx.copy()
    Hny = Hy.copy()

    # print("{:.6f}".format((np.square(En[1:nx-1,1:ny-1]-E_a[1:nx-1,1:ny-1])).mean(axis=None)))

    # print("{:.6f}".format((np.square(Hnx[1:nx-1,0:ny-1]-Hx_a[1:nx-1,0:ny-1])).mean()))
    # print("{:.6f}".format((np.square(Hny[0:nx-1,1:ny-1]-Hy_a[0:nx-1,1:ny-1])).mean(axis=None)))
    # print("{:.6f}".format(abs(En-E_a).max()))

    E[1:nx - 1, 1:ny - 1] = En[1:nx - 1, 1:ny - 1] + (Z * dt / dx) * (
                Hny[1:nx - 1, 1:ny - 1] - Hny[0:nx - 2, 1:ny - 1]) - (Z * dt / dy) * (
                                        Hnx[1:nx - 1, 1:ny - 1] - Hnx[1:nx - 1, 0:ny - 2])

    Em = E.copy()
    Hx[1:nx - 1, 0:ny - 1] = Hnx[1:nx - 1, 0:ny - 1] - (dt / (Z * dy)) * (Em[1:nx - 1, 1:ny] - Em[1:nx - 1, 0:ny - 1])
    Hy[0:nx - 1, 1:ny - 1] = Hny[0:nx - 1, 1:ny - 1] + (dt / (Z * dx)) * (Em[1:nx, 1:ny - 1] - Em[0:nx - 1, 1:ny - 1])
    # q=(np.square(((((Hx[1:nx,0:ny-1]-Hx[0:nx-1,0:ny-1])+(Hy[0:nx-1,1:ny]-Hy[0:nx-1,0:ny-1]))/(dx)).max()))).mean()
    # print("{:.6f}".format(q) )

    # print(np.square(Em))
    loss_E += np.sqrt((np.square(E - E_a[n + 1])).mean(axis=None))
    loss_H += np.sqrt((np.square(Hx[1:nx - 1, 0:ny - 1] - Hx_a[n + 1][1:nx - 1, 0:ny - 1])).mean(axis=None))
    loss_H += np.sqrt((np.square(Hy[0:nx - 1, 1:ny - 1] - Hy_a[n + 1][0:nx - 1, 1:ny - 1])).mean(axis=None))
    Loss_E.append(loss_E.copy())
    Loss_H.append(loss_H.copy())
    # print(E.max())
    # print('E_error='+"{:.6f}".format(np.sqrt((np.square(E-E_a[n+1])).mean(axis=None))))
    # print('Hx_error='+"{:.6f}".format(np.sqrt((np.square(Hx[1:nx-1,0:ny-1]-Hx_a[n+1][1:nx-1,0:ny-1])).mean(axis=None))))
    # print('Hy_error='+"{:.6f}".format(np.sqrt((np.square(Hy[0:nx-1,1:ny-1]-Hy_a[n+1][0:nx-1,1:ny-1])).mean(axis=None))))

plt.figure()
plt.plot(np.arange(time_steps_2) * dt, Loss_E, color='red', label='Error for E')
plt.plot(np.arange(time_steps_2) * dt, Loss_H, color='black', label='Error for H')
plt.xlabel('time')
plt.ylabel('error')
plt.legend()
plt.figure()
plt.show()

# plt.plot(Hx_a[n+1][:,10],color='red')
# plt.plot(E_a[0][:,20],color='red')
# print(E_a[0][10,10])
generate_html(dt,c,xmax,nx,time_steps_2,Hx_a,tot_hx,"Hx.html")
