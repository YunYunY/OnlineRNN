import torch

# define the inputs
x = [Variable(FloatTensor([i]), requires_grad=True) for i in (1, 2, 1)]
x1, x2, x3 = x

# define W and U as constant
weights = [Variable(FloatTensor([i]), requires_grad=True) for i in (1, 2)]
W, U = weights


h1 = W * x1
h2 = U * h1 + W * x2
h3 = U * h2 + W * x3

L = h3

L.backward()
weight_names = ['W', 'U']
for index, weight in enumerate(weights):
    gradient, *_ = weight.grad.data
    weight_name = weight_names[index]
    print(f"Gradient of {weight_name} w.r.t to L: {gradient}")