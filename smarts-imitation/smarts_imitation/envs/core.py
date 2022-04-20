from typing import List
import numpy as np
import gym
from dataclasses import replace
from collections import deque
from typing import Dict

from envision.client import Client as Envision
from smarts.core.smarts import SMARTS
from smarts.core.scenario import Scenario
from smarts.core.traffic_history_provider import TrafficHistoryProvider

from smarts_imitation.utils import agent
from smarts_imitation.utils.common import subscribe_features
from smarts_imitation.utils.feature_group import FeatureGroup


class SMARTSImitation:
    def __init__(
        self,
        scenarios: List[str],
        traffic_name: None,
        action_range: np.ndarray,
        obs_stack_size: int = 1,
        vehicles: Dict = None,
        control_all_vehicles: bool = False,
        control_vehicle_num: int = 1,
        feature_type: str = "radius",
        closest_neighbor_num: int = 6,
        use_rnn: bool = False,
        envision: bool = False,
        envision_sim_name: str = None,
        envision_record_data_replay_path: str = None,
        headless: bool = False,
    ):
        self.feature_list = FeatureGroup[feature_type]
        self.control_all_vehicles = control_all_vehicles
        self.obs_stack_size = obs_stack_size
        self.use_rnn = use_rnn
        self.traffic_name = traffic_name

        self.control_vehicle_num = self.n_agents = control_vehicle_num
        self.vehicles = vehicles
        # print(self.vehicles)
        if vehicles is None:
            print("Use All Vehicles")

        self.scenarios_iterator = Scenario.scenario_variations(
            scenarios, [], shuffle_scenarios=False, circular=False
        )
        self._init_scenario()
        # Num of all combinations of different controlled vehicles used.
        self.episode_num = len(self.vehicle_ids) - self.control_vehicle_num + 1

        if self.control_all_vehicles:
            print("Control All Vehicles")
            assert vehicles is None
            # This must be called after self._init_scenario(), otherwise self.vehicle_ids is not initialized.
            self.control_vehicle_num = self.n_agents = len(self.vehicle_ids)
        self.aid_to_vid = {}
        self.agent_ids = [f"agent_{i}" for i in range(self.n_agents)]

        self.agent_spec = agent.get_agent_spec(self.feature_list, closest_neighbor_num)

        feature_spaces = subscribe_features(
            self.feature_list, closest_neighbor_num=closest_neighbor_num
        )
        feature_shape_sum = 0
        for _, space in feature_spaces.items():
            assert len(space.shape) == 1
            feature_shape_sum += space.shape[0]
        if self.use_rnn:
            observation_shape = (
                self.obs_stack_size,
                feature_shape_sum,
            )
        else:
            observation_shape = (self.obs_stack_size * feature_shape_sum,)

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=observation_shape,
            dtype=np.float64,
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float64
        )

        assert (
            action_range.shape == (2, 2) and (action_range[1] >= action_range[0]).all()
        ), action_range
        self._action_range = action_range  # np.array([[low], [high]])

        envision_client = None
        if envision:
            envision_client = Envision(
                sim_name=envision_sim_name,
                output_dir=envision_record_data_replay_path,
                headless=headless,
            )

        self.smarts = SMARTS(
            agent_interfaces={},
            traffic_sim=None,
            envision=envision_client,
        )

        if obs_stack_size > 1:
            self.obs_queue_n = {
                a_id: deque(maxlen=obs_stack_size) for a_id in self.agent_ids
            }

    def seed(self, seed):
        np.random.seed(seed)

    def _convert_obs(self, raw_observations):
        full_obs_n = {}
        for agent_id in raw_observations.keys():
            observation = self.agent_spec.observation_adapter(
                raw_observations[agent_id]
            )
            all_states = []
            for feat in observation:
                all_states.append(observation[feat])
            full_obs = np.concatenate(all_states, axis=-1).reshape(-1)
            full_obs_n[agent_id] = full_obs
        return full_obs_n

    def step(self, action_n):
        scaled_action_n = {}
        for agent_id in action_n.keys():
            if self.done_n[agent_id]:
                continue
            action = action_n[agent_id]
            scaled_action = np.clip(action, -1, 1)
            # Transform the normalized action back to the original range
            # *** Formula for transformation from x in [xmin, xmax] to [ymin, ymax]
            # *** y = (ymax - ymin) * (x - xmin) / (xmax - xmin) + ymin
            scaled_action = (self._action_range[1] - self._action_range[0]) * (
                scaled_action + 1
            ) / 2 + self._action_range[0]
            scaled_action_n[agent_id] = self.agent_spec.action_adapter(scaled_action)

        raw_observation_n, reward_n, self.done_n, _ = self.smarts.step(scaled_action_n)
        full_obs_n = self._convert_obs(raw_observation_n)
        if self.obs_stack_size > 1:
            for agent_id in full_obs_n.keys():
                self.obs_queue_n[agent_id].append(full_obs_n[agent_id])
                if self.use_rnn:
                    full_obs_n[agent_id] = np.stack(
                        [obs for obs in list(self.obs_queue_n[agent_id])]
                    )
                else:
                    full_obs_n[agent_id] = np.concatenate(
                        [obs for obs in list(self.obs_queue_n[agent_id])], axis=-1
                    )

        info_n = {}
        for agent_id in full_obs_n.keys():
            info_n[agent_id] = {}
            info_n[agent_id]["reached_goal"] = raw_observation_n[
                agent_id
            ].events.reached_goal
            info_n[agent_id]["collision"] = (
                len(raw_observation_n[agent_id].events.collisions) > 0
            )
            info_n[agent_id]["car_id"] = self.aid_to_vid[agent_id]
            info_n[agent_id]["raw_position"] = raw_observation_n[
                agent_id
            ].ego_vehicle_state.position

        if self.time_slice:
            for agent_id in full_obs_n.keys():
                vehicle_id = self.aid_to_vid[agent_id]
                if self.smarts.elapsed_sim_time > self.vehicle_end_times[vehicle_id]:
                    self.done_n[agent_id] = True
                    print("time slice !!!!!!!!")

        return (
            full_obs_n,
            reward_n,
            self.done_n,
            info_n,
        )

    def reset(self):
        if self.episode_count == self.episode_num:
            self.episode_count = 0
            if self.control_vehicle_num > 1:
                self.vehicle_itr = np.random.choice(len(self.vehicle_ids))
            else:
                self.vehicle_itr = 0

        if self.vehicle_itr + self.n_agents > len(self.vehicle_ids):
            self.vehicle_itr = 0

        traffic_history_provider = self.smarts.get_provider_by_type(
            TrafficHistoryProvider
        )
        assert traffic_history_provider
        self.active_vehicle_ids = self.vehicle_ids[
            self.vehicle_itr : self.vehicle_itr + self.n_agents
        ]
        self.aid_to_vid = {
            f"agent_{i}": self.active_vehicle_ids[i] for i in range(self.n_agents)
        }

        agent_interfaces = {}
        # Find the earliest start time among all selected vehicles.
        history_start_time = np.inf
        for agent_id in self.agent_ids:
            vehicle_id = self.aid_to_vid[agent_id]
            agent_interfaces[agent_id] = self.agent_spec.interface
            history_start_time = min(
                history_start_time, self.vehicle_start_times[vehicle_id]
            )

        traffic_history_provider.start_time = history_start_time
        ego_missions = {}
        for agent_id in self.agent_ids:
            vehicle_id = self.aid_to_vid[agent_id]
            ego_missions[agent_id] = replace(
                self.vehicle_missions[vehicle_id],
                start_time=self.vehicle_start_times[vehicle_id] - history_start_time,
            )
        self.scenario.set_ego_missions(ego_missions)
        self.smarts.switch_ego_agents(agent_interfaces)

        raw_observation_n = self.smarts.reset(self.scenario)
        full_obs_n = self._convert_obs(raw_observation_n)
        if self.obs_stack_size > 1:
            for agent_id in full_obs_n.keys():
                self.obs_queue_n[agent_id].extend(
                    [full_obs_n[agent_id] for _ in range(self.obs_stack_size)]
                )
                if self.use_rnn:
                    full_obs_n[agent_id] = np.stack(
                        [obs for obs in list(self.obs_queue_n[agent_id])]
                    )
                else:
                    full_obs_n[agent_id] = np.concatenate(
                        [obs for obs in list(self.obs_queue_n[agent_id])],
                        axis=-1,
                    )

        self.done_n = {a_id: False for a_id in self.agent_ids}
        self.vehicle_itr += 1
        self.episode_count += 1
        return full_obs_n

    def _init_scenario(self):
        self.scenario = None
        for scenario in self.scenarios_iterator:
            if scenario._traffic_history.name == self.traffic_name:
                self.scenario = scenario
                break
        assert self.scenario is not None

        self.vehicle_missions = self.scenario.discover_missions_of_traffic_histories()

        if self.vehicles is None:
            self.time_slice = False
            self.vehicle_ids = list(self.vehicle_missions.keys())
            self.vehicle_start_times = {
                v_id: self.vehicle_missions[v_id].start_time
                for v_id in self.vehicle_ids
            }
        elif self.vehicles[0].start_time is None:
            self.time_slice = False
            self.vehicle_ids = [v.vehicle_id for v in self.vehicles]
            self.vehicle_start_times = {
                v_id: self.vehicle_missions[v_id].start_time
                for v_id in self.vehicle_ids
            }
        elif self.vehicles[0].start_time is not None:
            self.time_slice = True
            self.vehicle_ids = [v.vehicle_id for v in self.vehicles]
            self.vehicle_start_times = {
                v.vehicle_id: v.start_time for v in self.vehicles
            }
            self.vehicle_end_times = {v.vehicle_id: v.end_time for v in self.vehicles}

        # TODO(zbzhu): Need further discussion here, i.e., how to maintain BOTH randomness and sorted order of vehicles.
        if self.control_vehicle_num == 1:
            np.random.shuffle(self.vehicle_ids)
            self.vehicle_itr = 0
        else:
            # Sort vehicle id by starting time, so that we can get adjacent vehicles easily.
            self.vehicle_ids = self.vehicle_ids[np.argsort(self.vehicle_start_times)]
            self.vehicle_itr = np.random.choice(len(self.vehicle_ids))
        self.episode_count = 0

    def destroy(self):
        if self.smarts is not None:
            self.smarts.destroy()
