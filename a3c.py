from sc2gym import ACTIONS
from a2c import A2C
from env import Env


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

    def reset(self):

        state, aviliable_action_mask = self.env.reset()
        start_action, _ , _ = self.a2c.step(state, aviliable_action_mask)
        return start_action

    def run(self):

        t_start = 0
        t = 0
        rounds = 0
        while(rounds < TOTAL_ROUNDS):
            rounds += 1
            action = self.reset()
            timestamp = {}
            while(t < T_EVERY_ROUND):
                t += 1
                state, reward, done, aviliable_action_mask = self.env.step(
                    action)
                if done:
                    break
                action, V, policy = self.a2c.step(state, aviliable_action_mask)
                timestamp["reward"] += [reward]
                timestamp["V"] += [V]
                timestamp["policy"] += [policy]
                if t-t_start == DT:
                    t_start = t
                    R = timestamp["V"][-1]
                    for i in range(DT-1):
                        _t = -(i+2)
                        R = timestamp["reward"][_t] + GAMMA*R
                        V = timestamp["V"][_t]
                        policy = timestamp["policy"][_t]
                        loss = self.a2c.loss_funcion(R, V, policy)
                    timestamp = {}

if __name__ == "__main__":
    a3c = A3C(map_name="DefeatRoaches")
    start_action = a3c.reset()
    