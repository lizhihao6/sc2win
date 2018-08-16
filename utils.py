import logging
import torch
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def push_and_pull(opt, gnet, lnet, done, s, ns, buffer_s, buffer_ns, buffer_a, buffer_r, GAMMA):

    if done:
        V = 0.
    else:
        _, V = lnet.forward([s], [ns])

    buffer_v_target = []
    for R in buffer_r[::-1]:   # reverse buffer r
        V = R + GAMMA*float(V)
        buffer_v_target.append(V)
    buffer_v_target.reverse()

    loss = lnet.loss_function(buffer_s, buffer_ns, buffer_a, buffer_v_target)

    # calculate local gradients and push local parameters to global
    opt.zero_grad()
    loss.backward()
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = lp.grad
    opt.step()

    # pull global parameters
    lnet.load_state_dict(gnet.state_dict())


def recode(gnet, g_ep, ep_r, process_id):

    g_ep_v = 0
    with g_ep.get_lock():
        g_ep.value += 1
        g_ep_v = g_ep.value

    if g_ep_v % 100 == 0:
        torch.save(gnet.state_dict(), "model/params.pkl")

    logger.info("G_EP = %d, process %d got reward %d of ep_r",
                g_ep.value, process_id, ep_r)
