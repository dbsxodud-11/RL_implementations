import numpy as np
import torch

x = np.random.normal(0, 1, size=(3, 3))
x = torch.tensor(x, dtype=torch.float32)

y = torch.tensor([0, 2, 1]).unsqueeze(0)

z = torch.gather(x, 0, y)
print(x)
print(z)