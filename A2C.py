import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, screen_layer, minimap_layer, other_feature_lenght):

        super(Net, self).__init__()
        self.screen_conv1 = nn.Conv2d(screen_layer, 32, 3)
        self.screen_conv2 = nn.Conv2d(32, 32, 3)
        self.screen_adp_pool = nn.AdaptiveAvgPool2d(1)
        self.screen_fc1 = nn.Linear(32, 256)
        self.screen_fc2 = nn.Linear(256, 256)
        self.screen_fc3 = nn.Linear(256, 256)
        self.screen_fc4 = nn.Linear(256, 256)

        self.mini_conv1 = nn.Conv2d(minimap_layer, 32, 3)
        self.mini_conv2 = nn.Conv2d(32, 32, 3)
        self.mini_adp_pool = nn.AdaptiveAvgPool2d(1)
        self.mini_fc1 = nn.Linear(32, 256)
        self.mini_fc2 = nn.Linear(256, 256)
        self.mini_fc3 = nn.Linear(256, 256)
        self.mini_fc4 = nn.Linear(256, 256)

        self.other_fc1 = nn.Linear(256+256+other_feature_lenght, 256+256)
        self.other_fc2 = nn.Linear(256+256, 256)
        self.other_fc3 = nn.Linear(256, 256)
        self.other_fc4 = nn.Linear(256, 256)

        self.all_fc1 = nn.Linear(256+256+256, 256+256+256)
        self.all_fc2 = nn.Linear(256+256+256, 256+256+256)
        self.all_fc3 = nn.Linear(256+256+256, 524)
        self.all_fc4 = nn.Linear(524,524)
        self.all_fc5 = nn.Linear(524,524)

    def forward(self, screen_features, minimap_features, other_features):
        
        x = screen_features
        y = minimap_features
        z = other_features
        
        x = F.relu(self.screen_conv1(x))
        x = F.relu(self.screen_conv2(x))
        x = self.screen_adp_pool(x)
        x = F.relu(self.screen_fc1(x))
        x = F.relu(self.screen_fc2(x))
        x = F.relu(self.screen_fc3(x))
        x = F.relu(self.screen_fc4(x))

        y = F.relu(self.mini_conv1(y))
        y = F.relu(self.mini_conv2(y))
        y = self.mini_adp_pool(y)
        y = F.relu(self.mini_fc1(y))
        y = F.relu(self.mini_fc2(y))
        y = F.relu(self.mini_fc3(y))
        y = F.relu(self.mini_fc4(y))

        z = F.relu(self.other_fc1(z))
        z = F.relu(self.other_fc2(z))
        z = F.relu(self.other_fc3(z))
        z = F.relu(self.other_fc4(z))

        a = torch.cat((x,y,z),0)
        a = F.relu(self.all_fc1(a))
        a = F.relu(self.all_fc2(a))
        a = F.relu(self.all_fc3(a))
        a = F.relu(self.all_fc4(a))

        return a

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


if __name__ == "__main__":
    net = Net(13, 7,200)
    print(net)
