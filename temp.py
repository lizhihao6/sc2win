import torch
import torch.nn.functional as F
# li = [torch.Tensor([1,2]), torch.Tensor([3,4,9,0]), torch.Tensor([5,6,8])]
# print(torch.cat(li,0))
# entropy_loss = torch.nn.CrossEntropyLoss()
# a = torch.LongTensor([0])
# loss = entropy_loss(torch.Tensor([[1,2,3,5]]),a)
# print(loss)
a = torch.Tensor([1,3,3])
a = a.view(1,a.size()[0])
b = F.softmax(a, dim=1)
d = F.log_softmax(a,dim=1)
c = [-b_*d_ for b_,d_ in zip(b,d)][0]
e = torch.sum(c)
print(e)