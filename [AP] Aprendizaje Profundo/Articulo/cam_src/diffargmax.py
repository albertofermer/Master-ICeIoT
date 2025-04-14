from tkinter import X
import torch
import torch.nn as nn



l1 = nn.Linear(2, 4, bias=False)
with torch.no_grad():
    l1.weight = nn.Parameter(torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]]).float())

l2 = nn.Linear(4, 1, bias=False)
with torch.no_grad():
    l2.weight = nn.Parameter(torch.tensor([0, 1, 2, 3]).float())

input_vector = torch.tensor([0.3, 0.8])

x = input_vector
print(x)
x = l1(x)
print(x)
x = nn.functional.gumbel_softmax(x, tau=1, hard=True)
print(x)
x = l2(x)
print(x)

target = 1
loss = torch.abs(target - x)
loss.backward()
print(l1.weight.grad)