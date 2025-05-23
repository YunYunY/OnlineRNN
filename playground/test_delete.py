import torch
from torch.autograd import Variable


# define the inputs
x = [Variable(torch.FloatTensor([i]), requires_grad=True) for i in (1, 2, 1)]
x1, x2, x3 = x

# define W and U as constant
weights = [Variable(torch.FloatTensor([i]), requires_grad=True) for i in (1, 2)]
W, U = weights
print(x)
print(W)
h1 = W * x1 
h2 = U * h1 + W * x2
h3 = U * h2 + W * x3

L = W * x1
L2 = W * x2
W.require_grad = False
L3 = L + L2
L3.backward(retain_graph=True)
L.backward(retain_graph=True)

weight_names = ['W', 'U']
for index, weight in enumerate(weights):
    gradient, *_ = weight.grad.data
    weight_name = weight_names[index]
    print(f"Gradient of {weight_name} w.r.t to L: {gradient}")
    # weight.grad.data.zero_()  


L.backward(retain_graph=False)
# print(W.backward(W.grad, retain_graph=False))
weight_names = ['W', 'U']
for index, weight in enumerate(weights):
    gradient, *_ = weight.grad.data
    weight_name = weight_names[index]
    print(f"Gradient of {weight_name} w.r.t to L: {gradient}")