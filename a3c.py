from sc2gym import ACTIONS
from a2c import A2C
from env import Env
import torch

TOTAL_ROUNDS = 4000
T_EVERY_ROUND = 200
DT = 10
GAMMA = 0.9


class A3C:

    def __init__(self, **kwargs):

        self.env = Env(**kwargs)
        state, _ = self.env.reset()
        self.a2c = A2C(len(state[0]), len(state[1]), len(
            state[2]), self.env.action_length, self.env.args_length_dict)

    def run(self):

        opt = torch.optim.Adam(self.a2c.parameters(), lr=1e-7, betas=(0.9, 0.9), eps=1e-8,weight_decay=0)

        rounds = 0
        while(rounds < TOTAL_ROUNDS):
            t_start = 0
            loss = 0
            t = 0
            rounds += 1
            state, aviliable_action_mask = self.env.reset()
            timestamp = {"reward": [], "V": [], "action_for_policy": []}
            while(t < T_EVERY_ROUND):
                t += 1
                action, V, policy, action_for_policy = self.a2c.step(
                    state, aviliable_action_mask)
                state, reward, done, aviliable_action_mask = self.env.step(
                    action)
                if done:
                    break
                timestamp["reward"].append(reward)
                timestamp["V"].append(V)
                timestamp["action_for_policy"].append(action_for_policy)
                if (t-t_start) == DT:
                    t_start = t
                    R = float(timestamp["V"][-1])
                    for i in range(DT-1):
                        _t = -(i+2)
                        R = float(timestamp["reward"][_t]) + GAMMA*R
                        V = timestamp["V"][_t]
                        action_for_policy = timestamp["action_for_policy"][_t]
                        loss = self.a2c.loss_funcion(
                            R, float(V), policy, action_for_policy)
                        opt.zero_grad()
                        loss.backward(retain_graph=True)
                        opt.step()
            
            print(loss)

if __name__ == "__main__":
    a3c = A3C(map_name="DefeatRoaches")
    a3c.run()
