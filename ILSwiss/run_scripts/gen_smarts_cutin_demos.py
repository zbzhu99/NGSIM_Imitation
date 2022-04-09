import yaml
import argparse
import os
import sys
import pickle
import inspect
from pathlib import Path
from collections import defaultdict

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
print(sys.path)

import numpy as np
import math
from gen_smarts_demos import observation_transform, calculate_actions, \
    split_train_test

from rlkit.data_management.path_builder import PathBuilder
from rlkit.launchers.launcher_util import set_seed

from smarts.core.smarts import SMARTS
from smarts.core.scenario import Scenario
from smarts.core.utils.math import radians_to_vec, vec_to_radians
from smarts_imitation.utils import adapter, agent
from smarts_imitation import ScenarioZoo
from smarts_imitation.utils.common import _legalize_angle

lane_change_stats = defaultdict(int)

def sample_cutin_demos(train_vehicle_ids, scenarios, specs):
    done_vehicle_num = 0
    agent_spec = agent.get_agent_spec(
        specs["env_specs"]["env_kwargs"]["feature_list"],
        specs["env_specs"]["env_kwargs"]["closest_neighbor_num"],
    )
    observation_adapter = adapter.get_observation_adapter(
        feature_list=specs["env_specs"]["env_kwargs"]["feature_list"],
        closest_neighbor_num=specs["env_specs"]["env_kwargs"]["closest_neighbor_num"],
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
    all_original_paths = {}
    cutin_demo_trajs = []
    total_cutin_steps = 0

    if specs["env_specs"]["env_kwargs"]["obs_stack_size"] > 1:
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
        raw_observations,
        observation_adapter,
        obs_queues,
        specs["env_specs"]["env_kwargs"]["obs_stack_size"],
        specs["env_specs"]["env_kwargs"]["use_rnn"],
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
            specs["env_specs"]["env_kwargs"]["obs_stack_size"],
            specs["env_specs"]["env_kwargs"]["use_rnn"],
        )
        actions = calculate_actions(raw_observations, raw_next_observations)

        """ Handle terminated vehicles. """
        for vehicle in done_vehicles:
            if vehicle.split("-")[-1] in train_vehicle_ids:
                done_vehicle_num += 1
                vehicle_name = "Agent-" + vehicle
                all_original_paths[vehicle_name][-1]["terminals"] = True

                original_path = all_original_paths.pop(vehicle_name)
                cutin_demo_traj, cutin_steps = get_single_cutin_demo(
                    original_path, specs
                )
                if cutin_demo_traj is not None:
                    cutin_demo_trajs.append(cutin_demo_traj)
                    total_cutin_steps += cutin_steps
                print(f"Agent-{vehicle} Ended, total {done_vehicle_num} Ended. "
                      f"Cutin Demo Trajs: {len(cutin_demo_trajs)}, "
                      f"curr steps: {cutin_steps}, "
                      f"Total cutin steps: {total_cutin_steps}",
                      f"curr_stats: {lane_change_stats}")

        """ Store data in the corresponding path builder. """
        vehicles = next_observations.keys()

        for vehicle in vehicles:
            if vehicle.split("-")[-1] in train_vehicle_ids and vehicle in observations:
                if vehicle not in all_original_paths:
                    all_original_paths[vehicle] = []

                all_original_paths[vehicle].append(
                    {
                        "observations": observations[vehicle],
                        "actions": actions[vehicle],
                        "rewards": np.array([0.0]),
                        "next_observations": next_observations[vehicle],
                        "terminals": np.array([False]),
                        "raw_observations": raw_observations[vehicle],
                        "raw_next_observations": raw_next_observations[vehicle],
                    }
                )

        raw_observations = raw_next_observations
        observations = next_observations

    return cutin_demo_trajs


def _is_lane_change_valid(raw_observations, angle_threshold, cutin_dist_threshold):
    """Args:
    raw_observations: raw_observation of ego vehicle after lane change.
    """
    ego = raw_observations.ego_vehicle_state
    neighbor_vehicle_states = raw_observations.neighborhood_vehicle_states
    for v in neighbor_vehicle_states:
        if v.lane_index != ego.lane_index:
            continue

        rel_pos_vec = np.array(
            [v.position[0] - ego.position[0], v.position[1] - ego.position[1]]
        )
        rel_pos_radians = vec_to_radians(rel_pos_vec)
        angle = _legalize_angle(rel_pos_radians - ego.heading)
        # The neighbor vehicle is behind of the ego vehicle.
        if angle < angle_threshold or angle > 2 * math.pi - angle_threshold:
            continue
        dist = np.linalg.norm(v.position[:2] - ego.position[:2], 2)
        if dist < cutin_dist_threshold:
            continue
        return True

    return False


def _lane_change_steps(original_path, specs):
    global lane_change_stats
    lane_change_steps = []
    for i in range(len(original_path)):
        cur_lane = original_path[i]["raw_observations"].ego_vehicle_state.lane_index
        next_lane = original_path[i][
            "raw_next_observations"
        ].ego_vehicle_state.lane_index
        if next_lane != cur_lane:
            if _is_lane_change_valid(
                original_path[i]["raw_next_observations"],
                specs["env_specs"]["env_kwargs"]["angle_threshold"],
                specs["env_specs"]["env_kwargs"]["cutin_dist_threshold"],
            ):
                lane_change_steps.append(i)
                lane_change_stats[(cur_lane, next_lane)] += 1
    return lane_change_steps


def should_be_save(index, lane_change_steps, steps_before_cutin, steps_after_cutin):
    for step in lane_change_steps:
        if step - steps_before_cutin <= index <= step + steps_after_cutin:
            return True
    return False


def get_single_cutin_demo(original_path, specs):

    cutin_steps = 0
    lane_change_steps = _lane_change_steps(original_path, specs)
    if len(lane_change_steps) == 0:  # no lane change
        return None, 0
    cur_path_builder = PathBuilder(["agent_0"])
    for i in range(len(original_path)):
        if should_be_save(
            i,
            lane_change_steps,
            specs["env_specs"]["env_kwargs"]["steps_before_cutin"],
            specs["env_specs"]["env_kwargs"]["steps_after_cutin"],
        ):
            cur_path_builder["agent_0"].add_all(
                observations=original_path[i]["observations"],
                actions=original_path[i]["actions"],
                rewards=original_path[i]["rewards"],
                next_observations=original_path[i]["next_observations"],
                terminals=original_path[i]["terminals"],
            )
            cutin_steps += 1

    return cur_path_builder, cutin_steps


def experiment(specs):

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
    cutin_demo_trajs = sample_cutin_demos(
        train_vehicle_ids,
        ScenarioZoo.get_scenario("NGSIM-I80"),
        specs,
    )

    print(
        "\nOBS STACK SIZE: {}\n".format(
            specs["env_specs"]["env_kwargs"]["obs_stack_size"]
        )
    )

    file_save_path = Path(save_path).joinpath(
        "smarts_{}_stack-{}_cutin.pkl".format(
            exp_specs["env_specs"]["scenario_name"],
            exp_specs["env_specs"]["env_kwargs"]["obs_stack_size"],
        )
    )
    with open(
        file_save_path,
        "wb",
    ) as f:
        pickle.dump(cutin_demo_trajs, f)

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
