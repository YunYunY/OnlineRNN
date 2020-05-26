import torch
a = torch.tensor(1., requires_grad=True)
b = 2 * (a.detach())
# states = [(None, a)]

# state = states[-1][1].detach()
# state.requires_grad=True
# b = 2 * a
# new_state = 2 * state
# states.append((state, new_state))

# state = states[-1][1].detach()
# state.requires_grad=True
# new_state = a + state
# states.append((state, new_state))
c = a + b

# state = states[-1][1].detach()
# state.requires_grad=True
# new_state = 2 * state
# states.append((state, new_state))
d = 2 * c
# print(states[0][1])
# print(torch.autograd.grad(states[-1][1], states[0][-1], only_inputs=True, retain_graph=True))
print(torch.autograd.grad(d, a, only_inputs=True, retain_graph=True))
