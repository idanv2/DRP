#import numpy as np
#import math
import torch
import torch.nn.functional as F
from main1 import amper, faraday,run_p
#import tensorflow as tf
#run_p()
#
# nx=5
# ny=5
# c=5.
# n=1
# dt=0.1
# P=3.14
# k1=2.
# k2=3.
# dx=1/(nx-1)
# dy=1/(ny-1)
# E_a=np.zeros((nx,ny))
# for i in range(0, nx ):
#     for j in range(0, ny):
#         E_a[i, j] =math.sin(dx*i)+(dy*j)**2
#
# x=np.linspace(0,1,nx)
# y=x
# X, Y=np.meshgrid(x,y,indexing='ij')
# E= np.sin(X)+(Y)**2
# # print(E-E_a)
E=np.array([[1., 2.],[3., 4.]])
E=torch.tensor(E).float().reshape(2,1,2)
inputs = torch.randn(2, 1, 2)
filters = torch.randn(1, 1, 2)
filters=torch.tensor([1.,-1.]).reshape(1,1,2)
print(F.conv1d(E, filters))
#print(E.dtype)
#print(filters.requires_grad)
#
# import torch
# def f(x,y):
#     return x**2+y
# x=torch.tensor(1.,device=device,requires_grad=True)
# y=torch.tensor(2.,device=device)
# #print(x.requires_grad)
# #with torch.no_grad():
# loss1=f(x,y)
# loss2=f(x,y)*3
# loss=loss1*loss2
# loss.backward()
# print(x.grad)
#
#











