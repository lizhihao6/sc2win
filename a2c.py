import torch
import torch.nn as nn
import torch.nn.functional as F
from sc2gym import ACTIONS

ENT_COEF = 1e-3
VF_COEF = 1


class A2C(nn.Module):

    def __init__(self, space_feature_dict: dict, nospace_featuce: list, action_dict: dict):

        super(A2C, self).__init__()

        self.space_name = space_feature_dict.keys()

        for name in self.space_name:
            self.cnn(name)
            self.cnn(name + "_v")

        self.liner("nospace_feature", len(nospace_featuce), 256)

        input_dim = (len(self.space_name) + 1) * 256
        self.action_dict = action_dict

        for name in action_dict.keys():

            output_dim = len(action_dict[name])
            self.liner(name + "_output", input_dim, output_dim)

        self.liner("value_output", input_dim, 1)

    def cnn(self, name: str):

        self.cnn_name_list = ["_conv1", "_conv2", "_adp_pool", "_fc1", "_fc2"]
        setattr(self, name + "_conv1", nn.Conv2d(1, 16, kernel_size=8, stride=4))
        setattr(self, name + "_conv2",
                nn.Conv2d(16, 32, kernel_size=4, stride=2))
        setattr(self, name + "_adp_pool", nn.AdaptiveAvgPool2d(1))
        setattr(self, name + "_fc1", nn.Linear(32, 256))
        setattr(self, name + "_fc2", nn.Linear(256, 256))

    def cnn_forward(self, x: torch.Tensor, name: str):

        name_list = [name + cnn_name for cnn_name in self.cnn_name_list]
        conv1 = getattr(self, name_list[0])
        conv2 = getattr(self, name_list[1])
        adp_pool = getattr(self, name_list[2])
        fc1 = getattr(self, name_list[3])
        fc2 = getattr(self, name_list[4])

        x = F.relu(conv1(x))
        x = F.relu(conv2(x))
        x = adp_pool(x)
        x = F.relu(fc1(x.view(1, -1)))
        x = fc2(x)

        return x

    def liner(self, name: str, input_dim: int, output_dim: int):

        setattr(self, name + "_fc1", nn.Linear(input_dim, 1024))
        setattr(self, name + "_fc2", nn.Linear(1024, output_dim))

    def liner_forward(self, x: torch.Tensor, name: str):

        fc1 = getattr(self, name + "_fc1")
        fc2 = getattr(self, name + "_fc2")

        x = F.relu(fc1(x))

        return fc2(x)

    def forward(self, space_feature_dict: dict, nospace_featuce: list):

        self.train()

        a_list = []

        for name in self.space_name:
            a = torch.Tensor([[space_feature_dict[name]]]).cuda()
            a_list.append(self.cnn_forward(a, name))

        a = torch.Tensor([nospace_featuce]).cuda()
        a_list.append(self.liner_forward(a, "nospace_feature"))

        a_tensor = torch.cat(a_list, -1)

        policy = [self.liner_forward(a_tensor, name + "_output")
                  for name in self.action_dict.keys()]
        value = self.liner_forward(a_tensor.view(1, -1), "value_output")

        return policy, value

    def choose_action(self, policy: torch.Tensor, action_id_mask: list) -> dict:

        self.eval()

        action = {}
        mask_tensor = torch.ByteTensor(action_id_mask).cuda()
        index_list = []
        for i, index in zip(action_id_mask, range(len(action_id_mask))):
            if i == 1:
                index_list.append(index)

        for args, name in zip(policy, self.action_dict):

            if name == "action_id":
                args = torch.masked_select(args, mask_tensor)

            probs = F.softmax(args, dim=0)
            m = torch.distributions.Categorical(probs)
            if name == "action_id":
                action_id = index_list[int(m.sample())]
                action_id = torch.Tensor([action_id])
                action[name] = action_id.cuda()
            else:
                action[name] = m.sample()

        return action

    def loss_function(self, R: float, V: torch.Tensor, policy: list, action: dict) -> torch.Tensor:

        self.train()

        A = R - V
        critic_loss = VF_COEF*A.pow(2)

        actor_loss = torch.Tensor([[0]]).cuda()

        for args, action_name in zip(policy, action.keys()):

            probs = F.softmax(args, dim=1)
            m = torch.distributions.Categorical(probs)
            actor_loss -= m.log_prob(action[action_name]) * A.detach()

        entropy_loss = 0.0
        entropy_loss_fn = nn.CrossEntropyLoss()
        for args, action_name in zip(policy, action.keys()):
            entropy_loss += entropy_loss_fn(args, action[action_name].long())

        return (critic_loss + actor_loss/20 + ENT_COEF*entropy_loss).mean()
