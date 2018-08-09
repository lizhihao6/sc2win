import torch
import torch.nn as nn
import torch.nn.functional as F
from sc2gym import ACTIONS


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
        action_ids, V, args_dict = self.forward(features)
        prob = F.softmax(action_ids, dim=1).data
        aviliable_actions_mask = torch.ByteTensor(aviliable_actions_mask)
        prob = torch.masked_select(prob, aviliable_actions_mask)
        m = self.distribution(prob)
        acition_id = m.sample()
        action.append(acition_id)
        args_name_list = ACTIONS._ARGS[acition_id]
        for arg_name in args_dict.keys():
            if arg_name in args_name_list:
                if arg_name in ["screen", "screen2", "minimap"]:
                    x_prob = F.softmax(args_dict[arg_name][0], dim=1)
                    y_prob = F.softmax(args_dict[arg_name][1], dim=1)
                    x_m = self.distribution(x_prob)
                    y_m = self.distribution(y_prob)
                    action.append([x_m.sample(), y_m.sample()])
                else:
                    prob = F.softmax(args_dict[arg_name], dim=1)
                    m = self.distribution(prob)
                    action.append(m.sample())

        policy = []
        policy.append(action_ids)
        for arg_name in args_dict.keys():
            if arg_name in ["screen", "screen2", "minimap"]:
                policy.append(args_dict[arg_name][0])
                policy.append(args_dict[arg_name][1])
            else:
                policy.append(args_dict[arg_name])

        return action, V, policy

    def loss_funcion(self, R, V, policy):

        A = R - V
        critic_loss = V.pow(2)

        probs = F.softmax(policy, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * A.detach()
        actor_loss = -exp_v

        entropy_loss = nn.CrossEntropyLoss(policy)

        total_loss = (critic_loss + action_loss + entropy_loss).mean()
        return total_loss


if __name__ == "__main__":
    from sc2gym import ACTIONS
    args_length_dict = ACTIONS._ARGS_MAX
    args_length_dict["screen"] = (64, 64)
    args_length_dict["screen2"] = (64, 64)
    args_length_dict["minimap"] = (32, 32)

    net = A2C(13, 7, 200, 524, args_length_dict)

    print(net)
