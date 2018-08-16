import torch
import torch.nn as nn
import torch.nn.functional as F
from sc2gym import ACTIONS

ENT_COEF = 1e-3
VF_COEF = 0.25


class A2C(nn.Module):

    def __init__(self, space_feature_dict: dict, nospace_featuce: list, action_dict: dict):

        super(A2C, self).__init__()

        self.space_name = [str(name) for name in space_feature_dict.keys()]

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
        x = F.relu(fc1(x.view(x.size(0), -1)))
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

    """
    To reduce the backward time, forward can input several state, and output several policys and values

    s_list = [dict1{"space_feactename": [], ...}, dict2, ...]
    policys = [action_id_policy:tensor[[time0],[time1],...], args_1_policy:tensor[[],[],...], ...]    
    values = tensor:[[time0],[time1],...]
    """
    def forward(self, s_list: list, ns_list: list):

        all_list = []

        for name in self.space_name:
            s = [[s_dict[name]] for s_dict in s_list]
            s = torch.Tensor(s).cuda()
            all_list.append(self.cnn_forward(s, name))

        s = torch.Tensor(ns_list).cuda()
        all_list.append(self.liner_forward(s, "nospace_feature"))
        all_tensor = torch.cat(all_list, -1)

        policys = [self.liner_forward(all_tensor, name + "_output")
                  for name in self.action_dict.keys()]

        values = self.liner_forward(all_tensor, "value_output")

        return policys, values

    """
    Because the action_id_mask is single, so choose_action can only input a policy, not policys.
    """
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
                action_id = torch.Tensor([action_id]).cuda()
                action[name] = action_id    
            else:
                action[name] = m.sample()

        return action

    def loss_function(self, buffer_s:list, buffer_ns:list, buffer_a:list, buffer_v_target:list) -> torch.Tensor:

        self.train()

        policys, values = self.forward(buffer_s, buffer_ns)
        b_f_t = torch.Tensor(buffer_v_target).cuda()
        A = b_f_t - values[0] # output is [[value1, value2, value3]]
        critic_loss = VF_COEF*A.pow(2)

        actor_loss = torch.Tensor([[0]]).cuda()

        action_dict = {}
        for name in self.action_dict.keys():
            action_dict[name] = [a[name] for a in buffer_a] 
        a_l = len(buffer_a)

        for args, name in zip(policys, self.action_dict.keys()):
            for arg,i in zip(args, range(a_l)):
                probs = F.softmax(arg, dim=0)
                m = torch.distributions.Categorical(probs)
                actor_loss -= m.log_prob(action_dict[name][i]) * A.detach()[i]
        actor_loss = actor_loss/len(policys[0])

        entropy_loss = 0.0
        entropy_loss_fn = nn.CrossEntropyLoss()
        for args, name in zip(policys, self.action_dict.keys()):
            for arg, i in zip(args, range(a_l)):
                entropy_loss += entropy_loss_fn(torch.unsqueeze(arg, 0), action_dict[name][i].long())
        entropy_loss = ENT_COEF*entropy_loss/len(policys[0])
        
        return (critic_loss + actor_loss + entropy_loss).mean()
