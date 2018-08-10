import sc2gym
import numpy as np
from sc2gym import ACTIONS


class Env:


    def __init__(self, **kwargs):

        self.env = sc2gym.SC2GameEnv(**kwargs)
        self.screen_size = self.env.screen_size
        self.minimap_size = self.env.minimap_size
        self.used_actions = self.action_filter([i for i in range(524)])
        self.action_length = len(self.used_actions)
        self.args_length_dict = ACTIONS._ARGS_MAX
        self.args_length_dict["screen"] = self.screen_size
        self.args_length_dict["screen2"] = self.screen_size
        self.args_length_dict["minimap"] = self.minimap_size


    def reset(self):
        
        state, _ = self.env.reset()
        return self.state2numpy(state), self.action2numpy(state["available_actions"])


    def step(self, action):
        
        state, reward, done, _ = self.env.step(action)
        return self.state2numpy(state), reward, done, self.action2numpy(state["available_actions"])


    def action2numpy(self, available_actions):

        actions_array = np.zeros(len(self.used_actions))
        actions_array[self.action_filter(available_actions)] = 1
        return actions_array


    def state2numpy(self, state):

        state = self.state_filter(state)

        screen_list = []
        minimap_list = []
        others_list = []

        for key in state.keys():
            if key in sc2gym.SCREEN_FEATURES._NAMES:
                screen_list.append(state[key])
            elif key.replace('mini_', '') in sc2gym.MINIMAP_FEATURES._NAMES:
                minimap_list.append(state[key])
            elif key == "available_actions" or key == "available_actions_args_max":
                pass
            else:
                array = np.array(state[key]).reshape(-1)
                for i in array:
                    others_list.append(i)

        screen_array = np.array(screen_list)
        minimap_array = np.array(minimap_list)
        others_array = np.array(others_list)

        return [screen_array, minimap_array, others_array]


    def state_filter(self, state):

        # # if you want to use a part of state, edit the name_list
        # name_list = [
        #     # feature layer 的 state
        #     "height_map", "visibility_map", "creep", "power", "player_id", "player_relative", "unit_type", "selected", "unit_hit_points", "unit_hit_points_ratio", "unit_energy", "unit_energy_ratio", "unit_shields", "unit_shields_ratio", "unit_density", "unit_density_aa", "effects",

        #     # minimap feature layer 的 state
        #     "mini_height_map", "mini_visibility_map", "mini_creep", "mini_camera", "mini_player_id", "mini_player_relative", "mini_selected",

        #     # 其余state
        #     "multi_select",
        #     "build_queue",
        #     "available_actions",
        #     "available_actions_args_max",
        #     "player",
        #     "game_loop",
        #     "score_cumulative",
        #     "single_select",
        #     "control_groups"
        # ]
        # _state = {}
        # for name in state.keys():
        #     if name in name_list:
        #         _state[name] = state[name]
        # return _state
        name_list = [
            # feature layer 的 state
            "player_id", "player_relative", "unit_type", "selected", "unit_hit_points", "unit_density", "unit_density_aa", 

            # minimap feature layer 的 state
            "mini_player_id", "mini_player_relative", "mini_selected",
            # 其余state
            "multi_select",
            "available_actions",
            "available_actions_args_max",
            "player",
            "game_loop",
            "single_select",
            "control_groups"
        ]
        _state = {}
        for name in state.keys():
            if name in name_list:
                _state[name] = state[name]
        return _state
        # return state


    def action_filter(self, actions):

        # if you want to use a part of state, edit the used_actions
        # used_actions = [i for i in range(524)]
        # _actions = []
        # for act in actions:
        #     if act in used_actions:
        #         _actions.append(used_actions.index(act))
        # return _actions
        # return actions
        # if you want to use a part of state, edit the used_actions
        used_actions = [i for i in range(524)]
        _actions = []
        for act in actions:
            if act in used_actions:
                _actions.append(used_actions.index(act))
        return _actions

def test():
    a = np.zeros(8)
    a[1, 3] = 1
    return a


if __name__ == "__main__":
    env = Env(map_name="CollectMineralsAndGas")
    print(env.minimap_size)
    # state_list, available_actions = env.reset()
    # print(available_actions)
    # print(state_list[0].shape)
    # print(state_list[1].shape)
    # print(state_list[2])
