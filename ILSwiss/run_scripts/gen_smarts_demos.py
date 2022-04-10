import yaml
import argparse
import os
import sys
import pickle
import inspect
from pathlib import Path
from collections import deque

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
print(sys.path)

import numpy as np
import time
from copy import deepcopy

from rlkit.data_management.path_builder import PathBuilder
from rlkit.launchers.launcher_util import set_seed

from smarts.core.smarts import SMARTS
from smarts.core.scenario import Scenario
from smarts.core.utils.math import radians_to_vec
from smarts_imitation.utils import adapter, agent
from smarts_imitation import ScenarioZoo


def split_train_test(scenarios, test_ratio):
    scenario_iterator = Scenario.scenario_variations(
        [scenarios],
        list([]),
    )
    scenario = next(scenario_iterator)
    vehicle_missions = scenario.discover_missions_of_traffic_histories()
    vehicle_ids = list(vehicle_missions.keys())
    np.random.shuffle(vehicle_ids)

    test_vehicle_ids = vehicle_ids[: int(len(vehicle_ids) * test_ratio)]
    train_vehicle_ids = vehicle_ids[int(len(vehicle_ids) * test_ratio) :]

    return train_vehicle_ids, test_vehicle_ids


def convert_single_obs(single_observation, observation_adapter):
    observation = observation_adapter(single_observation)
    all_states = []
    for feat in observation:
        all_states.append(observation[feat])
    full_obs = np.concatenate(all_states, axis=-1).reshape(-1)
    return full_obs


def observation_transform(
    raw_observations, observation_adapter, obs_queues, obs_stack_size, use_rnn
):
    observations = {}
    for vehicle_id in raw_observations.keys():
        if obs_stack_size > 1:
            converted_single_obs = convert_single_obs(
                raw_observations[vehicle_id], observation_adapter
            )
            if vehicle_id not in obs_queues.keys():
                obs_queues[vehicle_id] = deque(maxlen=obs_stack_size)
                obs_queues[vehicle_id].extend(
                    [deepcopy(converted_single_obs) for _ in range(obs_stack_size)]
                )
            else:
                obs_queues[vehicle_id].append(converted_single_obs)
            if not use_rnn:
                observations[vehicle_id] = np.concatenate(
                    list(obs_queues[vehicle_id]),
                    axis=-1,
                )
            else:
                observations[vehicle_id] = np.stack(list(obs_queues[vehicle_id]))
        else:
            observations[vehicle_id] = convert_single_obs(
                raw_observations[vehicle_id], observation_adapter
            )
    return observations


def calculate_actions(raw_observations, raw_next_observations, dt=0.1):
    actions = {}
    for car in raw_observations.keys():
        if car not in raw_next_observations.keys():
            continue
        car_next_state = raw_next_observations[car].ego_vehicle_state
        acceleration = car_next_state.linear_acceleration[:2].dot(
            radians_to_vec(car_next_state.heading)
        )
        angular_velocity = car_next_state.yaw_rate
        actions[car] = np.array([acceleration, angular_velocity])
    return actions


def sample_demos(
    train_vehicle_ids,
    scenarios,
    obs_stack_size,
    feature_list,
    closest_neighbor_num,
    use_rnn,
):
    agent_spec = agent.get_agent_spec(feature_list, closest_neighbor_num)
    observation_adapter = adapter.get_observation_adapter(
        feature_list, closest_neighbor_num
    )

    smarts = SMARTS(
        agent_interfaces={},
        traffic_sim=None,
    )
    scenarios_iterator = Scenario.scenario_variations(
        [scenarios],
        list([]),
    )

    prev_vehicles = set()
    path_builders = {}
    demo_trajs = []

    if obs_stack_size > 1:
        obs_queues = {}
    else:
        obs_queues = None

    """ Reset environment. """
    smarts.reset(next(scenarios_iterator))
    smarts.step({})
    smarts.attach_sensors_to_vehicles(
        agent_spec.interface, smarts.vehicle_index.social_vehicle_ids()
    )
    raw_observations, _, _, dones = smarts.observe_from(
        smarts.vehicle_index.social_vehicle_ids()
    )
    observations = observation_transform(
        raw_observations, observation_adapter, obs_queues, obs_stack_size, use_rnn
    )

    while True:
        """Step in the environment."""
        smarts.step({})

        current_vehicles = smarts.vehicle_index.social_vehicle_ids()
        done_vehicles = prev_vehicles - current_vehicles
        prev_vehicles = current_vehicles

        if len(current_vehicles) == 0:
            break

        smarts.attach_sensors_to_vehicles(
            agent_spec.interface, smarts.vehicle_index.social_vehicle_ids()
        )
        raw_next_observations, _, _, dones = smarts.observe_from(
            smarts.vehicle_index.social_vehicle_ids()
        )
        next_observations = observation_transform(
            raw_next_observations,
            observation_adapter,
            obs_queues,
            obs_stack_size,
            use_rnn,
        )
        actions = calculate_actions(raw_observations, raw_next_observations)

        """ Handle terminated vehicles. """
        for vehicle in done_vehicles:
            if vehicle.split("-")[-1] in train_vehicle_ids:
                cur_path_builder = path_builders["Agent-" + vehicle]
                cur_path_builder["agent_0"]["terminals"][-1] = True
                demo_trajs.append(cur_path_builder)
                print(f"Agent-{vehicle} Ended, total {len(demo_trajs)} Ended.")

        """ Store data in the corresponding path builder. """
        vehicles = next_observations.keys()

        for vehicle in vehicles:
            if vehicle.split("-")[-1] in train_vehicle_ids and vehicle in observations:
                if vehicle not in path_builders:
                    path_builders[vehicle] = PathBuilder(["agent_0"])

                path_builders[vehicle]["agent_0"].add_all(
                    observations=observations[vehicle],
                    actions=actions[vehicle],
                    rewards=np.array([0.0]),
                    next_observations=next_observations[vehicle],
                    terminals=np.array([False]),
                )

        raw_observations = raw_next_observations
        observations = next_observations

    return demo_trajs


def experiment(specs):
    time_start = time.time()
    save_path = Path("./demos/ngsim")
    os.makedirs(save_path, exist_ok=True)

    if not os.path.exists(save_path / "train_ids.pkl") or not os.path.exists(
        save_path / "test_ids.pkl"
    ):
        print(
            "\nSplit training and testing vehicles, with test ratio {}\n".format(
                specs["test_ratio"]
            )
        )
        train_vehicle_ids, test_vehicle_ids = split_train_test(
            ScenarioZoo.get_scenario("NGSIM-I80"),
            specs["test_ratio"],
        )

        with open(save_path / "train_ids.pkl", "wb") as f:
            print(f"Train Vehicle Num: {len(train_vehicle_ids)}")
            pickle.dump(train_vehicle_ids, f)
        with open(save_path / "test_ids.pkl", "wb") as f:
            print(f"Test Vehicle Num: {len(test_vehicle_ids)}")
            pickle.dump(test_vehicle_ids, f)

    else:
        with open(save_path / "train_ids.pkl", "rb") as f:
            train_vehicle_ids = pickle.load(f)
        print(f"Loading Train Vehicle Num: {len(train_vehicle_ids)}")

    # obtain demo paths
    demo_trajs = sample_demos(
        train_vehicle_ids,
        ScenarioZoo.get_scenario("NGSIM-I80"),
        specs["env_specs"]["env_kwargs"]["obs_stack_size"],
        feature_list=specs["env_specs"]["env_kwargs"]["feature_list"],
        closest_neighbor_num=specs["env_specs"]["env_kwargs"]["closest_neighbor_num"],
        use_rnn=specs["env_specs"]["env_kwargs"]["use_rnn"],
    )

    print(
        "\nOBS STACK SIZE: {}\n".format(
            specs["env_specs"]["env_kwargs"]["obs_stack_size"]
        )
    )

    with open(
        Path(save_path).joinpath(
            "smarts_{}_stack-{}.pkl".format(
                exp_specs["env_specs"]["scenario_name"],
                exp_specs["env_specs"]["env_kwargs"]["obs_stack_size"],
            ),
        ),
        "wb",
    ) as f:
        pickle.dump(demo_trajs, f)
    total_time = time.time() - time_start
    print("total time: {:2f}s".format(total_time))
    return 1


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", help="experiment specification file")
    parser.add_argument("-g", "--gpu", help="gpu id", type=int, default=0)
    args = parser.parse_args()
    with open(args.experiment, "r") as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string, Loader=yaml.FullLoader)

    set_seed(exp_specs["seed"])

    experiment(exp_specs)
