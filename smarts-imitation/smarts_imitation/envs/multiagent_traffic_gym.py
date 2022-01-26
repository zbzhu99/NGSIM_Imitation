import numpy as np
import gym
from dataclasses import replace
from pathlib import Path

from smarts.core.smarts import SMARTS
from smarts.core.scenario import Scenario
from smarts.core.traffic_history_provider import TrafficHistoryProvider

from smarts_imitation.utils import agent


class MASMARTSImitation(gym.Env):
    def __init__(self, scenarios, action_range, agent_number, vehicle_ids=None):
        super(MASMARTSImitation, self).__init__()
        self.scenarios_iterator = Scenario.scenario_variations(scenarios, [])
        self._next_scenario()
        self.obs_stacked_size = 1
        self.n_agents = agent_number
        self.agentid_to_vehid = {}
        self.agent_ids = [f"agent_{i}" for i in range(self.n_agents)]
        self.agent_spec = agent.get_agent_spec(self.obs_stacked_size)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(28,), dtype=np.float64
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float64
        )

        if vehicle_ids is not None:
            self.vehicle_ids = vehicle_ids
        else:
            print("Use ALL vehicles")

        assert (
            action_range.shape == (2, 2) and (action_range[1] >= action_range[0]).all()
        ), action_range
        self._action_range = action_range  # np.array([[low], [high]])

        self.smarts = SMARTS(
            agent_interfaces={},
            traffic_sim=None,
            envision=None,
        )

    def seed(self, seed):
        np.random.seed(seed)

    def change_agents_n(self, agent_number):
        self.n_agents = agent_number
        self.agent_ids = [f"agent_{i}" for i in range(self.n_agents)]

    def _convert_obs(self, raw_observations):
        full_obs_n = {}
        for agent_id in raw_observations.keys():
            observation = self.agent_spec.observation_adapter(
                raw_observations[agent_id]
            )
            ego_state = []
            other_info = []
            for feat in observation:
                if feat in ["ego_pos", "speed", "heading"]:
                    ego_state.append(observation[feat])
                else:
                    other_info.append(observation[feat])
            ego_state = np.concatenate(ego_state, axis=1).reshape(-1)
            other_info = np.concatenate(other_info, axis=1).reshape(-1)
            full_obs = np.concatenate((ego_state, other_info))
            full_obs_n[agent_id] = full_obs
        return full_obs_n

    def step(self, action):
        action_n = {}
        for agent_id in action.keys():
            if self.dones[agent_id]:
                continue
            agent_action = action[agent_id]
            agent_action = np.clip(agent_action, -1, 1)
            # Transform the normalized action back to the original range
            # *** Formula for transformation from x in [xmin, xmax] to [ymin, ymax]
            # *** y = (ymax - ymin) * (x - xmin) / (xmax - xmin) + ymin
            agent_action = (self._action_range[1] - self._action_range[0]) * (
                agent_action + 1
            ) / 2 + self._action_range[0]
            action_n[agent_id] = self.agent_spec.action_adapter(agent_action)
        raw_observations, rewards, dones, _ = self.smarts.step(action_n)
        self.dones = dones
        full_obs = self._convert_obs(raw_observations)
        info = {}
        for agent_id in full_obs.keys():
            info[agent_id] = {}
            info[agent_id]["reached_goal"] = raw_observations[
                agent_id
            ].events.reached_goal
            info[agent_id]["collision"] = (
                len(raw_observations[agent_id].events.collisions) > 0
            )

        return (
            full_obs,
            rewards,
            dones,
            info,
        )

    def reset(self):
        if self.vehicle_itr + self.n_agents >= (len(self.vehicle_ids) - 1):
            self._next_scenario()
        traffic_history_provider = self.smarts.get_provider_by_type(
            TrafficHistoryProvider
        )
        assert traffic_history_provider
        self.vehicle_id = self.vehicle_ids[
            self.vehicle_itr : self.vehicle_itr + self.n_agents
        ]

        for i in range(self.n_agents):
            self.agentid_to_vehid[f"agent_{i}"] = self.vehicle_id[i]

        agent_interfaces = {}
        history_start_time = self.vehicle_missions[self.vehicle_id[0]].start_time
        for agent_id in self.agent_ids:
            vehicle = self.agentid_to_vehid[agent_id]
            agent_interfaces[agent_id] = self.agent_spec.interface
            if history_start_time > self.vehicle_missions[vehicle].start_time:
                history_start_time = self.vehicle_missions[vehicle].start_time

        traffic_history_provider.start_time = history_start_time
        ego_missions = {}
        for agent_id in self.agent_ids:
            vehicle = self.agentid_to_vehid[agent_id]
            ego_missions[agent_id] = replace(
                self.vehicle_missions[vehicle],
                start_time=self.vehicle_missions[vehicle].start_time
                - history_start_time,
            )
        self.scenario.set_ego_missions(ego_missions)
        self.smarts.switch_ego_agents(agent_interfaces)

        observations = self.smarts.reset(self.scenario)
        full_obs = self._convert_obs(observations)
        self.dones = {}
        for agent_id in full_obs.keys():
            self.dones[agent_id] = False
        self.vehicle_itr += self.n_agents
        return full_obs

    def _next_scenario(self):
        self.scenario = next(self.scenarios_iterator)
        self.vehicle_missions = self.scenario.discover_missions_of_traffic_histories()
        self.veh_start_times = {
            v_id: mission.start_time for v_id, mission in self.vehicle_missions.items()
        }
        vlist = []
        for vehicle_id, start_time in self.veh_start_times.items():
            vlist.append((vehicle_id, start_time))
        dtype = [("id", int), ("start_time", float)]
        vlist = np.array(vlist, dtype=dtype)
        vlist = np.sort(vlist, order="start_time")
        self.vehicle_ids = list(self.vehicle_missions.keys())
        for id in range(len(self.vehicle_ids)):
            self.vehicle_ids[id] = f"{vlist[id][0]}"
        self.vehicle_itr = 0
        self.n_agents_max = len(self.vehicle_ids)

    def destroy(self):
        if self.smarts is not None:
            self.smarts.destroy()


if __name__ == "__main__":
    env = MASMARTSImitation(
        [str(Path(__file__).parent.parent.parent / "ngsim")],
        np.array([[0, 0], [1, 1]]),
        2,
    )
    obs = env.reset()
