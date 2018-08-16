from a2c import A2C
from env import Env
from sharedRMSprop import SharedRMSprop
from utils import push_and_pull, recode
import torch.multiprocessing as _mp
import torch,os, cloudpickle

mp = _mp.get_context('spawn')

TOTAL_ROUNDS = 20000
DT = 6
GAMMA = 0.99
PROCESS_NUM = 12

class A3C(mp.Process):

    def __init__(self, gnet, opt, global_ep, process_id):

        super(A3C, self).__init__()
        self.gnet = gnet
        self.opt, self.g_ep = opt, global_ep
        self.id = process_id

    def run(self):

        rounds = 0
        self.env = Env(map_name="DefeatRoaches")
        s, ns, action_dict = self.env.init()
        self.lnet = A2C(s, ns, action_dict).cuda()

        while(rounds < TOTAL_ROUNDS):

            t = t_start = 0
            rounds += 1
            s, ns = self.env.reset()
            buffer_s, buffer_ns, buffer_a, buffer_r = [], [], [], []
            ep_r = 0.

            while(True):

                t += 1
                policy, value = self.lnet.forward(
                    [s], [ns])
                buffer_s.append(s)
                buffer_ns.append(ns)

                action = self.lnet.choose_action(
                    policy, self.env.action_id_mask)  # maybe change read only
                buffer_a.append(action)

                s, ns, reward, done = self.env.step(
                    action)
                buffer_r.append(reward)
                ep_r += reward

                if (t-t_start) == DT or done:

                    t_start = t

                    push_and_pull(self.opt, self.gnet, self.lnet, done, s,
                                  ns, buffer_s, buffer_ns, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_ns, buffer_a, buffer_r = [], [], [], []

                    if done:
                        recode(self.gnet, self.g_ep, ep_r, self.id)
                        break


class GNET(mp.Process):

    def __init__(self):

        super(GNET, self).__init__()
        env = Env(map_name="DefeatRoaches")
        s, ns, action_dict = env.init()
        global gnet
        gnet = A2C(s, ns, action_dict)
        gnet.cuda()
        gnet.share_memory()
        return


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    global gnet
    GNET()

    if os.path.exists("model/params.pkl"):
        gnet.load_state_dict(torch.load("model/params.pkl"))
    
    opt = SharedRMSprop(gnet.parameters(), lr=1e-5)
    opt.share_memory()

    global_ep = mp.Value('i', 0)
    a3c_list = [A3C(gnet, opt, global_ep, i) for i in range(PROCESS_NUM)]
    [a3c.start() for a3c in a3c_list]
    [a3c.join() for a3c in a3c_list]
