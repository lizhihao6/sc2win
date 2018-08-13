from sc2gym import ACTIONS
from a2c import A2C
from env import Env
import torch

TOTAL_ROUNDS = 400
T_EVERY_ROUND = 200
DT = 10
GAMMA = 0.99


class A3C:

    def __init__(self, **kwargs):

        self.env = Env(**kwargs)
        space_feature_dict, nospace_feature, action_dict = self.env.init()
        self.a2c =A2C(space_feature_dict, nospace_feature, action_dict)
        self.a2c.cuda()

    def run(self):

        opt = torch.optim.Adam(self.a2c.parameters(), lr=1e-6, betas=(0.9, 0.9), eps=1e-8,weight_decay=0)

        rounds = 0
        
        while(rounds < TOTAL_ROUNDS):
        
            t_start = 0
            loss = None
            t = 0
            rounds += 1
            space_feature_dict, nospace_feature = self.env.reset()
            timestamp = {"reward": [], "value": [], "action": []}

            while(t < T_EVERY_ROUND):
        
                t += 1
                policy, value = self.a2c.forward(space_feature_dict, nospace_feature)
                action = self.a2c.choose_action(policy, self.env.action_id_mask) # maybe change read only
                space_feature_dict, nospace_feature, reward, done = self.env.step(action)

                timestamp["reward"].append(reward)
                timestamp["value"].append(value)
                timestamp["action"].append(action)
                
                if (t-t_start) == DT or done:

                    t_start = t
                    R = 0 if done else float(timestamp["value"][-1])
                    loss = 0.0

                    for i in range(len(timestamp["value"])-1):
                        
                        _t = -(i+2)
                        R = timestamp["reward"][_t] + GAMMA*R
                        V = timestamp["value"][_t]
                        action = timestamp["action"][_t]
                        loss += self.a2c.loss_function(R, V, policy, action)

                    opt.zero_grad()
                    loss.backward(retain_graph=True)
                    opt.step()

if __name__ == "__main__":
    a3c = A3C(map_name="MoveToBeacon")
    a3c.run()
