import torch
import numpy as np
epochs = 50
lr=0.000001
x=torch.tensor(1.)
w=torch.tensor(0.,requires_grad=True)
def g(x,w):
    return w*w+1
print(w.requires_grad)
for i in range(epochs):
        # Get the loss and perform backpropagation

        loss_f = g(x,w)
        #print(loss_f)
        loss_f.backward()
        # Let's update the weights
        with torch.no_grad():
            #print(w.grad)
            w-=w.grad*lr
            #Set the gradients to zero
            w.grad.zero_()

#print(w.grad)
print(g(x,w))