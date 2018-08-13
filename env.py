import sc2gym
import numpy as np
from sc2gym import ACTIONS


class Env:

    def __init__(self, **kwargs):

        self.env = sc2gym.SC2GameEnv(**kwargs)
        self.screen_size = self.env.screen_size
        self.minimap_size = self.env.minimap_size


    def init(self):

        state, _ = self.env.reset()
        space_feature_dict, nospace_feature = self.feature_wrap(state)
        action_dict = {"action_id":[0 for i in range(524)]}
        
        for name in ACTIONS._ARGS_MAX.keys():
        
            if name in ["screen", "screen2"]:
                action_dict[name+"_x"] = [0 for i in range(self.screen_size[0])]
                action_dict[name+"_y"] = [0 for i in range(self.screen_size[1])]
        
            elif name == "minimap":
                action_dict[name+"_x"] = [0 for i in range(self.minimap_size[0])]
                action_dict[name+"_y"] = [0 for i in range(self.minimap_size[1])]
        
            elif name in ["unload_id", "select_unit_id"]:
                action_dict[name] = [0 for i in range(10)]
        
            else:
                action_dict[name] = [0 for i in range(ACTIONS._ARGS_MAX[name][0])]
        
        return space_feature_dict, nospace_feature, action_dict

    def reset(self):

        state, _ = self.env.reset()

        return self.feature_wrap(state)

    def step(self, action_dict):

        action = self.action_unwrap(action_dict)
        state, reward, done, _ = self.env.step(action)
        space_feature_dict, nospace_feature = self.feature_wrap(state)

        return space_feature_dict, nospace_feature, float(reward), done

    def feature_wrap(self, state):

        space_feature_name_list = ["player_id", "player_relative", "unit_type", "selected",
                                   "unit_hit_points", "unit_hit_points_ratio", "unit_density", "unit_density_aa"]
        nospace_feature_name_list = [
            "multi_select", "player", "game_loop", "score_cumulative", "single_select", "control_groups"]

        space_feature_dict = {}
        nospace_feature = []
        self.action_id_mask = [0 for i in range(524)]

        for name in state.keys():

            if name in space_feature_name_list:
                space_feature_dict[name] = state[name]

            elif name in nospace_feature_name_list:
                nospace_feature += [i for i in np.array(state[name]).flatten()]

        for i in state["available_actions"]:
            self.action_id_mask[i] = 1

        return space_feature_dict, nospace_feature

    def action_unwrap(self, action_dict):

        action_id = int(action_dict["action_id"])
        action_args = ACTIONS._ARGS[action_id]
        action = [action_id]

        for arg in action_args:

            if arg in ["minimap", "screen", "screen2"]:
                x_y = (int(action_dict[arg+"_x"]), int(action_dict[arg+"_y"]))
                action.append(x_y)

            else:
                single_arg = (int(action_dict[arg]),)
                action.append(single_arg)

        return action


if __name__ == "__main__":

    env = Env(map_name="CollectMineralsAndGas")
