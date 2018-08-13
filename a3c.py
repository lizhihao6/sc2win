from sc2gym import ACTIONS
from a2c import A2C
from env import Env
from shared_adam import SharedAdam
import multiprocessing as mp
import threading
import torch
import os
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TOTAL_ROUNDS = 20000
DT = 16
GAMMA = 0.99
PROCESS_NUM = 16
global index


class A3C(threading.Thread):

    def __init__(self, **kwargs):

        threading.Thread.__init__(self)

        self._kwargs = kwargs

    def _init(self):
            
        self.env = Env(**self._kwargs)

        space_feature_dict, nospace_feature, action_dict = self.env.init()
        self.a2c = A2C(space_feature_dict, nospace_feature, action_dict)
        self.a2c.cuda()

    def run(self):

        self._init()

        global index
        self.id = index
        params_name = "model/params" + str(index) + ".pkl"
        if os.path.exists(params_name):
            self.a2c.load_state_dict(torch.load(params_name))

        opt = SharedAdam(self.a2c.parameters())

        rounds = 0

        while(rounds < TOTAL_ROUNDS):

            t_start = 0
            loss = None
            t = 0
            rounds += 1
            space_feature_dict, nospace_feature = self.env.reset()
            timestamp = {"reward": [], "value": [], "action": [], "space_feature_dict":[], "nospace_feature":[]}
            torch.save(self.a2c.state_dict(), params_name)

            while(True):

                t += 1
                policy, value = self.a2c.forward(
                    space_feature_dict, nospace_feature)
                timestamp["space_feature_dict"].append(space_feature_dict)
                timestamp["nospace_feature"].append(nospace_feature)
                
                action = self.a2c.choose_action(
                    policy, self.env.action_id_mask)  # maybe change read only
                timestamp["action"].append(action)

                space_feature_dict, nospace_feature, reward, done = self.env.step(
                    action)
                timestamp["reward"].append(reward)
                timestamp["value"].append(value)

                if (t-t_start) == DT or done:

                    t_start = t
                    R = 0 if done else float(timestamp["value"][-1])

                    for i in range(len(timestamp["value"])-1):

                        _t = -(i+2)
                        R = timestamp["reward"][_t] + GAMMA*R
                        V = timestamp["value"][_t]
                        action = timestamp["action"][_t]
                        space_feature_dict = timestamp["space_feature_dict"][_t]
                        nospace_feature = timestamp["nospace_feature"][_t]
                        self.loss = self.a2c.loss_function(
                            R, V, space_feature_dict, nospace_feature, action)

                        try:
                            opt.zero_grad()
                            self.loss.backward(retain_graph=True)
                            opt.step()
                        except:
                            print("Failed update weights with loss = ", self.loss)

                    timestamp = {"reward": [], "value": [], "action": [], "space_feature_dict":[], "nospace_feature":[]}

                    if done:
                        logger.info("id = %d, loss = %d", self.id, self.loss)
                        break


if __name__ == "__main__":

    mp.set_start_method('spawn')
    a3c_list = [A3C(map_name="DefeatRoaches") for i in range(PROCESS_NUM)]
    for a3c, i in zip(a3c_list, range(PROCESS_NUM)):
        global index
        index = i
        a3c.start()
    for a3c in a3c_list:
        a3c.join()
