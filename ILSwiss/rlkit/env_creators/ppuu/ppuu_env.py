import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent / "pytorch-PPUU"))
from map_i80_ctrl import ControlledI80

from rlkit.env_creators.base_env import BaseEnv


class PPUUEnv(BaseEnv):
    def __init__(self, vehicle_ids=None, **configs):
        super().__init__(**configs)

        # create underlying smarts simulator
        env_kwargs = configs["env_kwargs"]
        scenario_name = configs["scenario_name"]
        if scenario_name == "i80":
            self._env = ControlledI80(vehicle_ids=vehicle_ids, **env_kwargs)
        else:
            raise NotImplementedError(scenario_name)

        self._default_agent_name = "agent_0"
        self.agent_ids = [self._default_agent_name]
        self.n_agents = len(self.agent_ids)
        self.observation_space_n = {
            self._default_agent_name: self._env.observation_space
        }
        self.action_space_n = {self._default_agent_name: self._env.action_space}

    def get_unscaled_obs(self, obs):
        if self._env.normalise_state:
            # 7 = ego + 6 neighbors
            return (
                obs * (self._env.data_stats["s_std"].repeat(7).numpy() + 1e-8)
                + self._env.data_stats["s_mean"].repeat(7).numpy()
            )

    def __getattr__(self, attrname):
        return getattr(self._env, attrname)

    def seed(self, seed):
        return self._env.seed(seed)

    def reset(self):
        return {self._default_agent_name: self._env.reset()}

    def step(self, action_n):
        action = action_n[self._default_agent_name]
        next_obs, rew, done, _info = self._env.step(action)

        info = {}
        info["collision"] = _info["c"]
        info["reached_goal"] = _info["a"]
        info["car_id"] = _info["id"]

        next_obs_n = {self._default_agent_name: next_obs}
        rew_n = {self._default_agent_name: rew}
        done_n = {self._default_agent_name: done}
        info_n = {self._default_agent_name: info}
        return next_obs_n, rew_n, done_n, info_n

    def render(self, **kwargs):
        return self._env.render(**kwargs)
