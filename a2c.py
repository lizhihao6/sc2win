import torch
import torch.nn as nn
import torch.nn.functional as F
from sc2gym import ACTIONS

ENT_COEF = 1e-3
VF_COEF = 0.25

class A2C(nn.Module):

    def __init__(self, screen_layer, minimap_layer, other_feature_lenght, action_length, args_length_dict):

        super(A2C, self).__init__()

        self.args_length_dict = args_length_dict
        self.distribution = torch.distributions.Categorical

        # input screen_feature output 256 screen_liner
        self.screen_conv1 = nn.Conv2d(
            screen_layer, 16, kernel_size=8, stride=4)
        self.screen_conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.screen_adp_pool = nn.AdaptiveAvgPool2d(1)
        self.screen_fc1 = nn.Linear(32, 256)
        self.screen_fc2 = nn.Linear(256, 256)

        # input minimap_feature output 256 minimap_liner
        self.mini_conv1 = nn.Conv2d(minimap_layer, 16, 8, stride=4)
        self.mini_conv2 = nn.Conv2d(16, 32, 4, stride=2)
        self.mini_adp_pool = nn.AdaptiveAvgPool2d(1)
        self.mini_fc1 = nn.Linear(32, 256)
        self.mini_fc2 = nn.Linear(256, 256)

        # input other_feature output 256 other_liner
        self.other_fc1 = nn.Linear(other_feature_lenght, 256)
        self.other_fc2 = nn.Linear(256, 256)

        # input screen_liner + minimap_liner + other_liner output 256 state_liner
        self.all_fc1 = nn.Linear(256+256+256, 256)
        self.all_fc2 = nn.Linear(256, 256)

        # input state_liner output Value(state)
        self.V_fc1 = nn.Linear(256, 32)
        self.V_fc2 = nn.Linear(32, 1)

        # input state_liner output action_liner
        self.action_ids = nn.Linear(256, action_length)

        # input action_liner + state_liner output feature_liner
        for args_name in args_length_dict.keys():

            if args_name in ["minimap", "screen", "screen2"]:
                setattr(self, args_name+"_x_fc",
                        nn.Linear(256, args_length_dict[args_name][0]))
                setattr(self, args_name+"_y_fc",
                        nn.Linear(256, args_length_dict[args_name][1]))

            else:
                setattr(self, args_name+"_fc",
                        nn.Linear(256, args_length_dict[args_name][0]))

    def forward(self, features):

        x = torch.Tensor([features[0]])
        y = torch.Tensor([features[1]])
        z = torch.Tensor([features[2]])

        x = F.relu(self.screen_conv1(x))
        x = F.relu(self.screen_conv2(x))
        x = self.screen_adp_pool(x)
        x = F.relu(self.screen_fc1(x.view(1, -1)))
        x = F.relu(self.screen_fc2(x))

        y = F.relu(self.mini_conv1(y))
        y = F.relu(self.mini_conv2(y))
        y = self.mini_adp_pool(y)
        y = F.relu(self.mini_fc1(y.view(1, -1)))
        y = F.relu(self.mini_fc2(y))

        z = F.relu(self.other_fc1(z))
        z = F.relu(self.other_fc2(z))

        state = torch.cat((x, y, z), 0)
        state = state.view(1, -1)
        state = F.relu(self.all_fc1(state))
        state = self.all_fc2(state)

        V = F.relu(self.V_fc1(state))
        V = self.V_fc2(V)

        action_ids = self.action_ids(state)

        ald = self.args_length_dict
        args_dict = {}

        for args_name in ald.keys():

            if args_name in ["minimap", "screen", "screen2"]:

                fc = getattr(self, args_name+"_x_fc")
                x = fc(state)
                fc = getattr(self, args_name+"_y_fc")
                y = fc(state)
                args_dict[args_name] = [x, y]

            else:

                fc = getattr(self, args_name+"_fc")
                args = fc(state)
                args_dict[args_name] = args

        return action_ids, V, args_dict

    def step(self, features, aviliable_actions_mask):

        self.eval()
        action = []
        action_dict = {}
        action_for_policy = []
        policy = []
        action_ids, V, args_dict = self.forward(features)

        asm = aviliable_actions_mask
        index_list = []
        for index in range(len(asm)):
            if int(asm[index]) == 1:
                index_list.append(index)
        asm = torch.ByteTensor(asm)
        action_ids_msk = torch.masked_select(action_ids, asm)
        prob = F.softmax(action_ids_msk.view(1,action_ids_msk.size()[0]), dim=1).data
        m = self.distribution(prob)
        action_id = index_list[m.sample()]
        policy.append(action_ids)
        action_for_policy.append(torch.Tensor([action_id]))

        for arg_name in args_dict:
            if arg_name in ["screen", "screen2", "minimap"]:
                x_prob = F.softmax(args_dict[arg_name][0], dim=1)
                y_prob = F.softmax(args_dict[arg_name][1], dim=1)
                x_m = self.distribution(x_prob)
                y_m = self.distribution(y_prob)
                x = x_m.sample()
                y = y_m.sample()
                action_dict[arg_name] = [int(x), int(y)]
                action_for_policy.append(x)
                action_for_policy.append(y)
                policy.append(args_dict[arg_name][0])
                policy.append(args_dict[arg_name][1])
            else:
                prob = F.softmax(args_dict[arg_name], dim=1)
                m = self.distribution(prob)
                args = m.sample()
                action_dict[arg_name] = [int(args)]
                action_for_policy.append(args)
                policy.append(args_dict[arg_name])

        action.append(action_id)
        args_name_list = ACTIONS._ARGS[action_id]
        for arg_name in args_name_list:
            action.append(action_dict[arg_name])

        return action, V, policy, action_for_policy

    def loss_funcion(self, R, V, policy, action_for_policy):

        self.train()
        A = R - V
        A = torch.Tensor([A])
        critic_loss = VF_COEF*A.pow(2)

        exp_v_sum = 0.0
        for index in range(len(policy)):
            probs = F.softmax(policy[index], dim=1)
            m = self.distribution(probs)
            action = action_for_policy[index]
            exp_v_sum += m.log_prob(action) * A.detach()

        actor_loss = -exp_v_sum

        policy_tensor = torch.cat([p[0] for p in policy], 0)
        policy_tensor = policy_tensor.view(1, policy_tensor.size()[0])
        prob = F.softmax(policy_tensor, dim=1)
        prob_log = F.log_softmax(policy_tensor, dim=1)
        entropy_loss = -ENT_COEF * \
            torch.sum([-p*p_log for p, p_log in zip(prob, prob_log)][0])
        # entropy_loss = nn.CrossEntropyLoss(torch.cat(policy_list,0)).float
        # print(entropy_loss)

        total_loss = (critic_loss + actor_loss + float(entropy_loss)).mean()
        return total_loss


if __name__ == "__main__":
    from sc2gym import ACTIONS
    args_length_dict = ACTIONS._ARGS_MAX
    args_length_dict["screen"] = (64, 64)
    args_length_dict["screen2"] = (64, 64)
    args_length_dict["minimap"] = (32, 32)

    net = A2C(13, 7, 200, 524, args_length_dict)

    print(net)
