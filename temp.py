import torch
import torch.nn.functional as F
# li = [torch.Tensor([1,2]), torch.Tensor([3,4,9,0]), torch.Tensor([5,6,8])]
# print(torch.cat(li,0))
entropy_loss = torch.nn.CrossEntropyLoss()
a = torch.LongTensor([0,1])
a.cuda()
a[0] = 0
print(a)
# loss = entropy_loss(torch.Tensor([[1,2,3,5],[1,2,4,2]]),a)

# a = torch.Tensor([[1]])
# args = torch.Tensor([[3, 2, 4]])

# t = torch.ByteTensor([[1,0,1]])

# z = torch.masked_select(args, t)
# m = torch.distributions.Categorical(F.softmax(args,dim=0))
# print(m.sample())
# list_a = [1,2,2,4]
# list_b = []
# for i in list_a:
#     list_b.append(i if i==1 else None)
# print(list_b)