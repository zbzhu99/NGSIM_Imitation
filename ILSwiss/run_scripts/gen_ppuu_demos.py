import os
import sys
import yaml
import torch
import inspect
import pickle
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
print(sys.path)

from rlkit.data_management.path_builder import PathBuilder
from rlkit.launchers.launcher_util import set_seed


PPUU_DIR = Path(__file__).absolute().parent.parent.parent / "pytorch-PPUU"


def split_train_test(scenario_name, test_ratio):
    if scenario_name == "i80":
        data_dir = str(PPUU_DIR / "traffic-data/state-action-cost/data_i80_v0")
    else:
        raise NotImplementedError(scenario_name)

    # XXX(zbzhu): for now, only use the first time slot
    time_slots = [
        "trajectories-0400-0415",
        # "trajectories-0500-0515",
        # "trajectories-0515-0530",
    ]

    for time_slot_idx, time_slot in enumerate(time_slots):
        combined_data_path = f"{data_dir}/{time_slot}/all_data.pth"
        if os.path.isfile(combined_data_path):
            print(f"Loading data shard: {combined_data_path}")
            data = torch.load(combined_data_path)

            vehicle_ids = [
                int(file_id.split("/")[-1].split(".")[0][3:]) for file_id in data["ids"]
            ]
            np.random.shuffle(vehicle_ids)

            test_vehicle_ids = list(
                zip(
                    [time_slot_idx] * int(len(vehicle_ids) * test_ratio),
                    vehicle_ids[: int(len(vehicle_ids) * test_ratio)],
                )
            )
            train_vehicle_ids = list(
                zip(
                    [time_slot_idx]
                    * (len(vehicle_ids) - int(len(vehicle_ids) * test_ratio)),
                    vehicle_ids[int(len(vehicle_ids) * test_ratio) :],
                )
            )

        else:
            raise FileNotFoundError(combined_data_path)

    return train_vehicle_ids, test_vehicle_ids


def generate_demos(train_vehicle_ids, scenario_name: str, obs_stack_size: int = 1):

    if scenario_name == "i80":
        data_dir = str(PPUU_DIR / "traffic-data/state-action-cost/data_i80_v0")
    else:
        raise NotImplementedError(scenario_name)
    stats_path = data_dir + "/data_stats.pth"

    # XXX(zbzhu): for now, only use the first time slot
    time_slots = [
        "trajectories-0400-0415",
        # "trajectories-0500-0515",
        # "trajectories-0515-0530",
    ]

    print(f"Loading data stats: {stats_path}")
    stats = torch.load(stats_path)
    a_scale = stats.get("a_scale").numpy()
    # ego vehicle and 6 neighbor vehicles
    s_mean = stats.get("s_mean").repeat(7).numpy()
    s_std = stats.get("s_std").repeat(7).numpy()

    demo_trajs = []

    for time_slot_idx, time_slot in enumerate(time_slots):
        combined_data_path = f"{data_dir}/{time_slot}/all_data.pth"
        if os.path.isfile(combined_data_path):
            print(f"Loading data shard: {combined_data_path}")
            data = torch.load(combined_data_path)

            for idx, file_id in enumerate(data["ids"]):
                vehicle_id = int(file_id.split("/")[-1].split(".")[0][3:])
                if (time_slot_idx, vehicle_id) in train_vehicle_ids:
                    path_builder = PathBuilder(["agent_0"])
                    vehicle_states = data["states"][idx].numpy()
                    vehicle_actions = data["actions"][idx].numpy()

                    episode_length = vehicle_states.shape[0]
                    if episode_length >= obs_stack_size + 1:
                        for t in range(episode_length - obs_stack_size):
                            path_builder["agent_0"].add_all(
                                observations=vehicle_states[
                                    t : t + obs_stack_size
                                ].reshape(obs_stack_size, -1),
                                actions=vehicle_actions[t + 1 : t + obs_stack_size + 1],
                                rewards=np.array([0.0]),
                                next_observations=vehicle_states[
                                    t + 1 : t + obs_stack_size + 1
                                ].reshape(obs_stack_size, -1),
                                terminals=np.array([False]),
                            )
                        path_builder["agent_0"]["terminals"][-1] = True

                        path_builder["agent_0"]["actions"] = np.stack(
                            path_builder["agent_0"]["actions"]
                        )
                        path_builder["agent_0"]["observations"] = np.stack(
                            path_builder["agent_0"]["observations"]
                        )
                        path_builder["agent_0"]["next_observations"] = np.stack(
                            path_builder["agent_0"]["next_observations"]
                        )

                        """ Data Normalization """
                        # action scale
                        path_builder["agent_0"]["actions"] /= a_scale.reshape(1, 1, 2)
                        path_builder["agent_0"]["actions"] = path_builder["agent_0"][
                            "actions"
                        ][:, -1]

                        # obs standard normal
                        obs_shape = (1, 28)
                        path_builder["agent_0"]["observations"] -= s_mean.reshape(
                            *obs_shape
                        )
                        path_builder["agent_0"]["observations"] /= 1e-8 + s_std.reshape(
                            *obs_shape
                        )
                        path_builder["agent_0"]["observations"] = path_builder[
                            "agent_0"
                        ]["observations"].reshape(
                            path_builder["agent_0"]["observations"].shape[0], -1
                        )
                        path_builder["agent_0"]["next_observations"] -= s_mean.reshape(
                            *obs_shape
                        )
                        path_builder["agent_0"][
                            "next_observations"
                        ] /= 1e-8 + s_std.reshape(*obs_shape)
                        path_builder["agent_0"]["next_observations"] = path_builder[
                            "agent_0"
                        ]["next_observations"].reshape(
                            path_builder["agent_0"]["next_observations"].shape[0], -1
                        )

                        demo_trajs.append(path_builder)
                        print(f"Agent-{vehicle_id} Ended")
        else:
            raise FileNotFoundError(combined_data_path)

    return demo_trajs


def experiment(specs: Dict[str, Any]):

    save_path = Path("./demos/ppuu")
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    if not os.path.exists(save_path / "train_ids.pkl") or not os.path.exists(
        save_path / "test_ids.pkl"
    ):
        print(
            "\nSplit training and testing vehicles, with test ratio {}\n".format(
                specs["test_ratio"]
            )
        )
        train_vehicle_ids, test_vehicle_ids = split_train_test(
            specs["env_specs"]["scenario_name"],
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
    demo_trajs = generate_demos(
        train_vehicle_ids,
        specs["env_specs"]["scenario_name"],
        specs["env_specs"]["env_kwargs"]["nb_states"],
    )

    print(
        "\nOBS STACK SIZE: {}\n".format(specs["env_specs"]["env_kwargs"]["nb_states"])
    )

    with open(
        save_path.joinpath(
            "ppuu_{}_stack-{}.pkl".format(
                exp_specs["env_specs"]["scenario_name"],
                exp_specs["env_specs"]["env_kwargs"]["nb_states"],
            ),
        ),
        "wb",
    ) as f:
        pickle.dump(demo_trajs, f)

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
