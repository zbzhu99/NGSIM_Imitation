import yaml
import argparse
import os
import sys
import queue
import pickle
import inspect
from pathlib import Path
from multiprocessing import Process, Queue
from collections import defaultdict
from functools import partial

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
print(sys.path)

import numpy as np
import math
from gen_smarts_demos import observation_transform, calculate_actions, split_train_test

from rlkit.data_management.path_builder import PathBuilder
from rlkit.launchers.launcher_util import set_seed

from smarts.core.smarts import SMARTS
from smarts.core.scenario import Scenario
from smarts.core.utils.math import vec_to_radians
from smarts_imitation.utils import adapter, agent
from smarts_imitation import ScenarioZoo
from smarts_imitation.utils.common import _legalize_angle
from smarts_imitation.utils.feature_group import FeatureGroup
from smarts_imitation.utils.vehicle_info import VehicleInfo


def work_process(
    cutin_trajs_queue,
    scenario,
    obs_stack_size,
    feature_list,
    closest_neighbor_num,
    use_rnn,
    angle_threshold,
    cutin_dist_threshold,
    steps_before_cutin,
    steps_after_cutin,
):
    all_vehicle_ids = list(scenario.discover_missions_of_traffic_histories().keys())
    scenario_name = scenario.name
    traffic_name = scenario._traffic_history.name

    done_vehicle_num = 0
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

    prev_vehicles = set()
    all_original_paths = {}
    traffic_cutin_steps = 0

    if obs_stack_size > 1:
        obs_queues = {}
    else:
        obs_queues = None

    """ Reset environment. """
    smarts.reset(scenario)
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
        obs_stack_size,
        use_rnn,
    )
    elapsed_sim_time = smarts.elapsed_sim_time

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
        next_elapsed_sim_time = smarts.elapsed_sim_time
        actions = calculate_actions(raw_observations, raw_next_observations)

        """ Handle terminated vehicles. """
        for vehicle in done_vehicles:
            if vehicle.split("-")[-1] not in all_vehicle_ids:
                print(f"***************** vehicle {vehicle} not in all_vehicle_ids")
            vehicle_name = "Agent-" + vehicle
            if (
                vehicle.split("-")[-1] in all_vehicle_ids
                and vehicle_name in all_original_paths
            ):
                done_vehicle_num += 1
                all_original_paths[vehicle_name][-1]["terminals"] = True

                original_path = all_original_paths.pop(vehicle_name)
                (
                    cutin_demo_traj,
                    cutin_steps,
                    start_time,
                    end_time,
                ) = get_single_cutin_demo(
                    original_path,
                    angle_threshold,
                    cutin_dist_threshold,
                    steps_before_cutin,
                    steps_after_cutin,
                )
                if cutin_demo_traj is not None:
                    vehicle_id = vehicle.split("-")[-1]
                    vehicle_info = VehicleInfo(
                        vehicle_id=vehicle_id,
                        start_time=start_time,
                        end_time=end_time,
                        scenario_name=scenario_name,
                        traffic_name=traffic_name,
                    )
                    cutin_trajs_queue.put(
                        (
                            vehicle_info,
                            cutin_demo_traj,
                        )
                    )
                    traffic_cutin_steps += cutin_steps
                print(
                    f"Traffic: {traffic_name}: Agent-{vehicle} Ended, "
                    f"total {done_vehicle_num} Ended. "
                    f"cutin steps: {traffic_cutin_steps}",
                )

        """ Store data in the corresponding path builder. """
        vehicles = next_observations.keys()

        for vehicle in vehicles:
            if vehicle.split("-")[-1] in all_vehicle_ids and vehicle in observations:
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
                        "elapsed_sim_time": elapsed_sim_time,
                    }
                )

        raw_observations = raw_next_observations
        observations = next_observations
        elapsed_sim_time = next_elapsed_sim_time

    print(f"worker process: {traffic_name} finished!")


def sample_cutin_demos(
    scenarios,
    save_path,
    test_ratio,
    obs_stack_size,
    feature_list,
    closest_neighbor_num,
    use_rnn,
    angle_threshold,
    cutin_dist_threshold,
    steps_before_cutin,
    steps_after_cutin,
):
    scenario_iterator = Scenario.scenario_variations(
        scenarios, list([]), shuffle_scenarios=False, circular=False
    )  # scenarios with different traffic histories.

    worker_processes = []
    trajs_queue = Queue()  # Queues are process safe.
    for scenario in scenario_iterator:
        scenario_name = scenario.name
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
                angle_threshold,
                cutin_dist_threshold,
                steps_before_cutin,
                steps_after_cutin,
            ),
            daemon=True,
        )
        print(f"Scenario-{scenario_name}, Traffic-{traffic_name} start sampling.")
        p.start()
        worker_processes.append(p)

    # Don not call p.join().
    cutin_demo_trajs = {}
    cutin_vehicles = defaultdict(partial(defaultdict, list))
    while True:  # not reliable
        try:
            if len(cutin_demo_trajs) == 0:
                vehicle_info, traj = trajs_queue.get(block=True)
            else:
                vehicle_info, traj = trajs_queue.get(block=True, timeout=300)
            cutin_demo_trajs[vehicle_info] = traj
            cutin_vehicles[vehicle_info.scenario_name][
                vehicle_info.traffic_name
            ].append(vehicle_info)
            print(f"main process: collected cutin trajs num: {len(cutin_demo_trajs)}")
        except queue.Empty:
            print("Queue empty! stop collecting.")
            break
    print(f"Append to buffer finished! total {len(cutin_demo_trajs)} trajectories!")

    cutin_train_vehicles, cutin_test_vehicles = split_train_test(
        cutin_vehicles, test_ratio
    )

    with open(save_path / "train_vehicles_cutin.pkl", "wb") as f:
        pickle.dump(cutin_train_vehicles, f)
    with open(save_path / "test_vehicles_cutin.pkl", "wb") as f:
        pickle.dump(cutin_test_vehicles, f)

    cutin_train_demo_trajs = []
    for traffic_name, vehicles in cutin_train_vehicles.items():
        for vehicle in vehicles:
            cutin_train_demo_trajs.append(cutin_demo_trajs[vehicle])

    return cutin_train_demo_trajs


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


def _lane_change_steps(original_path, angle_threshold, cutin_dist_threshold):
    lane_change_steps = []
    for i in range(len(original_path)):
        cur_lane = original_path[i]["raw_observations"].ego_vehicle_state.lane_index
        next_lane = original_path[i][
            "raw_next_observations"
        ].ego_vehicle_state.lane_index
        if next_lane != cur_lane:
            if _is_lane_change_valid(
                original_path[i]["raw_next_observations"],
                angle_threshold,
                cutin_dist_threshold,
            ):
                lane_change_steps.append(i)
    return lane_change_steps


def should_be_save(index, lane_change_steps, steps_before_cutin, steps_after_cutin):
    for step in lane_change_steps:
        if step - steps_before_cutin <= index <= step + steps_after_cutin:
            return True
    return False


def get_single_cutin_demo(
    original_path,
    angle_threshold,
    cutin_dist_threshold,
    steps_before_cutin,
    steps_after_cutin,
):

    cutin_steps = 0
    lane_change_steps = _lane_change_steps(
        original_path, angle_threshold, cutin_dist_threshold
    )
    if len(lane_change_steps) == 0:  # no lane change
        return None, 0, None, None
    cur_path_builder = PathBuilder(["agent_0"])

    start_time = np.inf
    end_time = -np.inf
    for i in range(len(original_path)):
        if should_be_save(
            i,
            lane_change_steps,
            steps_before_cutin,
            steps_after_cutin,
        ):
            cur_path_builder["agent_0"].add_all(
                observations=original_path[i]["observations"],
                actions=original_path[i]["actions"],
                rewards=original_path[i]["rewards"],
                next_observations=original_path[i]["next_observations"],
                terminals=original_path[i]["terminals"],
            )
            cutin_steps += 1
            start_time = min(start_time, original_path[i]["elapsed_sim_time"])
            end_time = max(end_time, original_path[i]["elapsed_sim_time"])

    return cur_path_builder, cutin_steps, start_time, end_time


def experiment(specs):
    scenario_names = specs["env_specs"]["scenario_names"]
    save_path = Path(f"./demos/" + "_".join(scenario_names))
    os.makedirs(save_path, exist_ok=True)

    # obtain demo paths
    scenarios = [
        ScenarioZoo.get_scenario(scenario_name) for scenario_name in scenario_names
    ]
    cutin_demo_trajs = sample_cutin_demos(
        scenarios,
        save_path,
        test_ratio=specs["test_ratio"],
        obs_stack_size=specs["env_specs"]["env_kwargs"]["obs_stack_size"],
        feature_list=FeatureGroup[specs["env_specs"]["env_kwargs"]["feature_type"]],
        closest_neighbor_num=specs["env_specs"]["env_kwargs"]["closest_neighbor_num"],
        use_rnn=specs["env_specs"]["env_kwargs"]["use_rnn"],
        angle_threshold=specs["env_specs"]["env_kwargs"]["angle_threshold"],
        cutin_dist_threshold=specs["env_specs"]["env_kwargs"]["cutin_dist_threshold"],
        steps_before_cutin=specs["env_specs"]["env_kwargs"]["steps_before_cutin"],
        steps_after_cutin=specs["env_specs"]["env_kwargs"]["steps_after_cutin"],
    )

    print(
        "\nOBS STACK SIZE: {}\n".format(
            specs["env_specs"]["env_kwargs"]["obs_stack_size"]
        )
    )

    file_save_path = Path(save_path).joinpath(
        "smarts_{}_stack-{}_cutin.pkl".format(
            specs["env_specs"]["env_kwargs"]["feature_type"],
            specs["env_specs"]["env_kwargs"]["obs_stack_size"],
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
