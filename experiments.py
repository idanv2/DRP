import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from numpy import random

from IPython import display
import math
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

fig = plt.figure()
ax1 = plt.axes(xlim=(0., 1.), ylim=(-c*4,c*4))
line, = ax1.plot([], [], lw=2)
plt.xlabel('Longitude')
plt.ylabel('Latitude')

plotlays, plotcols = [2], ["black","red"]
lines = []
for index in range(2):
    lobj = ax1.plot([],[],lw=2,color=plotcols[index])[0]
    lines.append(lobj)


def init():
    for line in lines:
        line.set_data([],[])
    return lines

x1,y1 = [],[]
x2,y2 = [],[]

# fake data


def animate(n):
    x = np.linspace(0., xmax, nx)
    y = E_a[n][:,20]
    x1=x
    y1=y


    y = E_a[n][:,10]
    x2=x
    y2=y

    xlist = [x1, x2]
    ylist = [y1, y2]

    #for index in range(0,1):
    for lnum,line in enumerate(lines):
        line.set_data(xlist[lnum], ylist[lnum]) # set data for each line separately.

    return lines

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=time_steps_2, interval=80, blit=True)

video=anim.to_html5_video()
html=display.HTML(video)
display.display(html)
plt.close()
with open("data.html", "w") as file:
    file.write(video)
plt.show()