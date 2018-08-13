from sc2gym import ACTIONS
from a2c import A2C
from env import Env
import torch
import os

TOTAL_ROUNDS = 20000
DT = 16
GAMMA = 0.99


class A3C:

    def __init__(self, **kwargs):

        self.env = Env(**kwargs)
        space_feature_dict, nospace_feature, action_dict = self.env.init()
	
        self.a2c =A2C(space_feature_dict, nospace_feature, action_dict)
        self.a2c.cuda()
        if os.path.exists("params.pkl"):
                self.a2c.load_state_dict(torch.load("params.pkl"))

    def run(self):

        opt = torch.optim.Adam(self.a2c.parameters(), lr=1e-5, betas=(0.9, 0.9), eps=1e-8,weight_decay=0)

        rounds = 0
        
        while(rounds < TOTAL_ROUNDS):
        
            t_start = 0
            loss = None
            t = 0
            rounds += 1
            space_feature_dict, nospace_feature = self.env.reset()
            timestamp = {"reward": [], "value": [], "action": []}
            torch.save(self.a2c.state_dict(), "params.pkl")

            while(True):
        
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
                
                    for i in range(len(timestamp["value"])-1):
                        
                        _t = -(i+2)
                        R = timestamp["reward"][_t] + GAMMA*R
                        V = timestamp["value"][_t]
                        action = timestamp["action"][_t]
                        self.loss = self.a2c.loss_function(R, V, policy, action)

                        try:
                            opt.zero_grad()
                            self.loss.backward(retain_graph=True)
                            opt.step()
                        except:
                            print("Failed update weights with loss = ", loss)

                    timestamp = {"reward": [], "value": [], "action": []}

                    if done:
                        print(self.loss)
                        break

if __name__ == "__main__":
    a3c = A3C(map_name="DefeatRoaches")
    a3c.run()
