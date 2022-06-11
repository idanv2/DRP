import numpy as np
import math
import torch
def generate_data(xmax, nx,ny,k1,k2,dx,dy,dt,time_steps):
    c = math.pi * (np.sqrt(k1 ** 2 + k2 ** 2))
    P = math.pi
    x=np.linspace(0.,xmax,nx)
    X,Y =np.meshgrid(x, x, indexing='ij')
    E=[]
    Hx=[]
    Hy=[]
    for n in range(time_steps+1):
        E.append(torch.tensor( c * np.cos(c * n * dt) * (np.sin(P * k1 * X) * np.sin(P * k2 * Y) + np.sin(P * k2 * X) * np.sin(
            P * k1 * Y))))
        Hx.append(torch.tensor( np.sin(c * (dt / 2) * (2 * n + 1)) * (
                -P * k2 * np.sin(P * k1 * X) * np.cos(P * k2 * (Y + dy / 2)) - P * k1 * np.sin(
            P * k2 * X) * np.cos(P * k1 * (Y + dy / 2)))))
        Hy.append(torch.tensor( np.sin(c * (dt / 2) * (2 * n + 1)) * (
                P * k1 * np.cos(P * k1 * (X + dx / 2)) * np.sin(P * k2 * Y) + P * k2 * np.cos(
            P * k2 * (X + dx / 2)) * np.sin(P * k1 * Y))))
    return  [E,Hx,Hy]
#E_a,Hx_a,Hy_a=generate_data()
