import torch

X = torch.rand((5, 3), requires_grad=True, dtype=torch.float32) * (-1 - 1) + -1

