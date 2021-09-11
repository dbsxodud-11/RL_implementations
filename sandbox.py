import numpy as np
import torch
import torch.nn as nn

x = np.random.normal(0, 1, size=(3, 3))
x = torch.tensor(x, dtype=torch.float32)

y = torch.tensor([0, 2, 1]).unsqueeze(0)

z = torch.gather(x, 0, y)
print(x)
print(z)

x = torch.from_numpy(np.random.normal(0, 1, size=(10, 3)))
y = torch.from_numpy(np.random.normal(0, 1, size=(10, 1)))
z = torch.cat((x, y), dim=1)
print(z.shape)

x = torch.tensor([[1, 0]]).float().reshape(-1, 1)
y = torch.tensor([[1, 2]]).float().reshape(-1, 1)
mse_loss = nn.MSELoss()(x, y)

print(mse_loss)