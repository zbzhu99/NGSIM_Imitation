import yaml
import argparse
import os
import sys
import queue
import pickle
import inspect
from pathlib import Path
from collections import deque, OrderedDict, defaultdict

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
print(sys.path)

import numpy as np
import time
from copy import deepcopy
from multiprocessing import Process, Queue

from rlkit.data_management.path_builder import PathBuilder
from rlkit.launchers.launcher_util import set_seed

from smarts.core.smarts import SMARTS
from smarts.core.scenario import Scenario
from smarts.core.utils.math import radians_to_vec

from smarts_imitation.utils.feature_group import FeatureGroup
from smarts_imitation.utils import adapter, agent
from smarts_imitation import ScenarioZoo
from smarts_imitation.utils.vehicle_with_time import VehicleWithTime


def split_train_test(scenario_vehicles, test_ratio):
    scenario_vehicles = OrderedDict(sorted(scenario_vehicles.items()))
    train_vehicles = {}
    test_vehicles = {}

    total_trajs_num = sum([len(x) for x in scenario_vehicles.values()])
    test_trajs_num = int(total_trajs_num * test_ratio)
    for traffic_name, vehicles in scenario_vehicles.items():
        if 0 < test_trajs_num < len(vehicles):
            test_vehicles[traffic_name] = vehicles[:test_trajs_num]
            train_vehicles[traffic_name] = vehicles[test_trajs_num:]
            test_trajs_num = 0
        elif test_trajs_num > 0 and len(vehicles) <= test_trajs_num:
            test_vehicles[traffic_name] = vehicles
            test_trajs_num -= len(vehicles)
        elif test_trajs_num == 0:
            train_vehicles[traffic_name] = vehicles
        else:
            raise ValueError
    print(
        "train_vehicle_ids_number: {}".format(
            {key: len(vs) for key, vs in train_vehicles.items()}
        )
    )
    print(
        "test_vehicle_ids_number: {}".format(
            {key: len(vs) for key, vs in test_vehicles.items()}
        )
    )
    # keep order
    train_vehicles = OrderedDict(sorted(train_vehicles.items()))
    test_vehicles = OrderedDict(sorted(test_vehicles.items()))

    return train_vehicles, test_vehicles


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


def work_process(
    trajs_queue,
    scenario,
    obs_stack_size,
    feature_list,
    closest_neighbor_num,
    use_rnn,
):
    all_vehicle_ids = scenario.discover_missions_of_traffic_histories().keys()
    traffic_name = scenario._traffic_history.name

    agent_spec = agent.get_agent_spec(
        feature_list=feature_list,
        closest_neighbor_num=closest_neighbor_num,
    )
    observation_adapter = adapter.get_observation_adapter(
        feature_list=feature_list,
        closest_neighbor_num=closest_neighbor_num,
    )

    smarts = SMARTS(
        agent_interfaces={},
        traffic_sim=None,
    )

    """ Reset environment. """

    prev_vehicles = set()
    path_builders = {}

    if obs_stack_size > 1:
        obs_queues = {}
    else:
        obs_queues = None

    smarts.reset(scenario)
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
            if (
                vehicle.split("-")[-1] in all_vehicle_ids
                and "Agent-" + vehicle in path_builders
            ):
                cur_path_builder = path_builders["Agent-" + vehicle]
                cur_path_builder["agent_0"]["terminals"][-1] = True
                vehicle_id = vehicle.split("-")[-1]
                trajs_queue.put((traffic_name, vehicle_id, cur_path_builder))
                print(f"{traffic_name} Agent-{vehicle} Ended")

        """ Store data in the corresponding path builder. """
        vehicles = next_observations.keys()

        for vehicle in vehicles:
            if vehicle.split("-")[-1] in all_vehicle_ids and vehicle in observations:
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

    print(f"worker process: {traffic_name} finished!")


def sample_demos(
    scenarios,
    save_path,
    test_ratio,
    obs_stack_size,
    feature_list,
    closest_neighbor_num,
    use_rnn,
):
    scenario_iterator = Scenario.scenario_variations(
        [scenarios], list([]), shuffle_scenarios=False, circular=False
    )  # scenarios with different traffic histories.

    worker_processes = []
    trajs_queue = Queue()  # Queues are process safe.
    for scenario in scenario_iterator:
        traffic_name = scenario._traffic_history.name
        p = Process(
            target=work_process,
            args=(
                trajs_queue,
                scenario,
                obs_stack_size,
                feature_list,
                closest_neighbor_num,
                use_rnn,
            ),
            daemon=True,
        )
        print(f"Traffic {traffic_name} start sampling.")
        p.start()
        worker_processes.append(p)

    # Do not call p.join().
    demo_trajs = {}
    scenario_vehicles = defaultdict(list)
    while True:  # not reliable
        try:
            if len(demo_trajs) == 0:
                traffic_name, vehicle_id, traj = trajs_queue.get(block=True)
            else:
                traffic_name, vehicle_id, traj = trajs_queue.get(
                    block=True, timeout=100
                )
            vehicle_with_time = VehicleWithTime(
                vehicle_id=vehicle_id,
                start_time=None,
                end_time=None,
                traffic_name=traffic_name,
            )
            demo_trajs[vehicle_with_time] = traj
            scenario_vehicles[traffic_name].append(vehicle_with_time)
            print(f"main process: collected trajs num: {len(demo_trajs)}")
        except queue.Empty:
            print("Queue empty! stop collecting.")
            break
    print(f"Append to buffer finished! total {len(demo_trajs)} trajectories!")

    if not os.path.exists(save_path / "train_vehicles.pkl") or not os.path.exists(
        save_path / "test_vehicles.pkl"
    ):
        print(
            "\nSplit training and testing vehicles, with test ratio {}\n".format(
                test_ratio
            )
        )
        train_vehicles, test_vehicles = split_train_test(scenario_vehicles, test_ratio)

        with open(save_path / "train_vehicles.pkl", "wb") as f:
            pickle.dump(train_vehicles, f)
        with open(save_path / "test_vehicles.pkl", "wb") as f:
            pickle.dump(test_vehicles, f)

    else:
        with open(save_path / "train_vehicles.pkl", "rb") as f:
            train_vehicles = pickle.load(f)

    train_demo_trajs = []
    for traffic_name, vehicles in train_vehicles.items():
        for vehicle in vehicles:
            train_demo_trajs.append(demo_trajs[vehicle])

    return train_demo_trajs


def experiment(specs):
    time_start = time.time()
    scenario_name = specs["env_specs"]["scenario_name"]
    save_path = Path(f"./demos/{scenario_name}")
    os.makedirs(save_path, exist_ok=True)

    # obtain demo paths
    demo_trajs = sample_demos(
        ScenarioZoo.get_scenario(scenario_name),
        save_path,
        specs["test_ratio"],
        specs["env_specs"]["env_kwargs"]["obs_stack_size"],
        feature_list=FeatureGroup[specs["env_specs"]["env_kwargs"]["feature_type"]],
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
            "smarts_{}_{}_stack-{}.pkl".format(
                exp_specs["env_specs"]["scenario_name"],
                exp_specs["env_specs"]["env_kwargs"]["feature_type"],
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
