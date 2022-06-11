import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from matplotlib import animation
import torch
import math
from mpl_toolkits.mplot3d import axes3d
import matplotlib.animation as animation
from  matplotlib.animation import FuncAnimation
from IPython import display


# Grid parameters.
k1, k2 = 3., 3.
nx = 40  # number of points in the x direction
xmin, xmax = 0.0, 1.0  # limits in the x direction
ny = nx  # number of points in the y direction
ymin, ymax = 0.0, 1.0
# dt=0.0002
T = 1
time_steps = 80
dt = T / time_steps
time_steps_2 = time_steps*3   # limits in the y direction
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
tot_E=[]

for n in range(time_steps_2):
    loss_H = 0
    loss_E = 0
    tot_E.append(E.copy())
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
    Loss_E.append(loss_E)
    Loss_H.append(loss_H)

fig=plt.figure(1)
lines1=plt.plot([])
line1=lines1[0]
line2=lines1[0]

plt.xlim(0.,1.)
plt.ylim(-1.,1.)
def animate(frame):
    line1.set_data((x,tot_E[frame][:,20]))




anim=FuncAnimation(fig,animate,frames=time_steps_2,interval=100)
video=anim.to_html5_video()
html=display.HTML(video)
display.display(html)
plt.close()
with open("data.html", "w") as file:
    file.write(video)
#
#     # print('E_error='+"{:.6f}".format(np.sqrt((np.square(E-E_a[n+1])).mean(axis=None))))
#     # print('Hx_error='+"{:.6f}".format(np.sqrt((np.square(Hx[1:nx-1,0:ny-1]-Hx_a[n+1][1:nx-1,0:ny-1])).mean(axis=None))))
#     # print('Hy_error='+"{:.6f}".format(np.sqrt((np.square(Hy[0:nx-1,1:ny-1]-Hy_a[n+1][0:nx-1,1:ny-1])).mean(axis=None))))
# plt.figure()
# plt.plot(np.arange(time_steps_2) * dt, Loss_E, color='red', label='Error for E')
# plt.plot(np.arange(time_steps_2) * dt, Loss_H, color='black', label='Error for H')
# plt.xlabel('time')
# plt.ylabel('error')
# plt.legend()
# plt.figure()
# # plt.plot(Hx_a[n+1][:,10],color='red')
# # plt.plot(E_a[0][:,20],color='red')
# # print(E_a[0][10,10])
#
#
# fig = plt.figure()
# ax = axes3d.Axes3D(fig)
#
# # Initialize Bz when time = - dt / 2
# wframe = ax.plot_wireframe(X, Y, Hx, rstride=2, cstride=2)
#
# ax.plot_wireframe(X, Y, Hx_a[n + 1], rstride=2, cstride=2, color='red', linestyle='dashed')
# plt.show()
# print(c)
#
#
#
