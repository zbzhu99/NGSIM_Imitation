import gym
import smarts_imitation

from rlkit.env_creators.base_env import BaseEnv


class MASmartsEnv(BaseEnv):
    """A wrapper for gym Mujoco environments to fit in multi-agent apis."""

    def __init__(self, vehicle_ids=None, **configs):
        super().__init__(**configs)

        # create underlying smarts simulator
        scenario_name = configs["scenario_name"]
        env_kwargs = configs["env_kwargs"]
        if scenario_name == "ngsim":
            self._env = gym.make(
                "SMARTS-Imitation-v2", vehicle_ids=vehicle_ids, **env_kwargs
            )
        else:
            raise NotImplementedError

        self.default_agent_name = "agent_0"
        self.n_agents = self._env.n_agents
        self.agent_ids = self._env.agent_ids

        self.observation_space_n = dict(
            zip(
                self.agent_ids,
                [self._env.observation_space for _ in range(self._env.n_agents)],
            )
        )
        self.action_space_n = dict(
            zip(
                self.agent_ids,
                [self._env.action_space for _ in range(self._env.n_agents)],
            )
        )

    def seed(self, seed):
        return self._env.seed(seed)

    def reset(
        self,
    ):
        return self._env.reset()

    def step(self, action_n):
        next_obs, rew, done, info = self._env.step(action_n)

        return next_obs, rew, done, info

    def render(self, **kwargs):
        return self._env.render(**kwargs)
