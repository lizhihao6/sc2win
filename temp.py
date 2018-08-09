import torch
import torch.nn.functional as F
distribution = torch.distributions.Categorical
a = torch.Tensor([[1,3,4],[2,4,5]])
prob = F.softmax(a,dim=1)
m = distribution(prob)
print(m.sample())