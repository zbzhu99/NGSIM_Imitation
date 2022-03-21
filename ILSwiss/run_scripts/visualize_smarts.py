import yaml
import argparse
import joblib
import os
import sys
import gym
import inspect
import pickle
import numpy as np
import signal
import random
from subprocess import Popen

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
print(sys.path)

from rlkit.envs import get_env

import rlkit.torch.utils.pytorch_util as ptu
from rlkit.launchers.launcher_util import set_seed
from rlkit.envs.wrappers import ProxyEnv, NormalizedBoxActEnv, ObsScaledEnv
from rlkit.torch.common.policies import MakeDeterministic

from smarts_imitation import ScenarioZoo


def experiment(variant):
    with open("demos_listing.yaml", "r") as f:
        listings = yaml.load(f.read(), Loader=yaml.FullLoader)

    eval_split_path = listings[variant["expert_name"]]["train_split"][0]
    with open(eval_split_path, "rb") as f:
        eval_vehicle_ids = pickle.load(f)

    eval_vehicle_ids = np.array(eval_vehicle_ids)

    # Can specify vehicle ids to be visualized as follows.
    variant["num_vehicles"] = len(eval_vehicle_ids)
    print("Total Vehicle Num: ", len(eval_vehicle_ids))

    env_specs = variant["env_specs"]
    if env_specs["env_kwargs"].get("control_all_vehicles", False):
        eval_vehicle_ids = None
    env = get_env(env_specs, vehicle_ids=eval_vehicle_ids)
    env.seed(variant["seed"])

    print("\nEnv: {}: {}".format(env_specs["env_creator"], env_specs["scenario_name"]))
    print("kwargs: {}".format(env_specs["env_kwargs"]))
    print("Obs Space: {}".format(env.observation_space_n))
    print("Act Space: {}\n".format(env.action_space_n))

    env_wrapper = ProxyEnv  # Identical wrapper
    for act_space in env.action_space_n.values():
        if isinstance(act_space, gym.spaces.Box):
            env_wrapper = NormalizedBoxActEnv
            break

    if variant["scale_env_with_demo_stats"]:
        with open("demos_listing.yaml", "r") as f:
            listings = yaml.load(f.read(), Loader=yaml.FullLoader)
        demos_path = listings[variant["expert_name"]]["file_paths"][
            variant["expert_idx"]
        ]

        print("demos_path", demos_path)
        with open(demos_path, "rb") as f:
            traj_list = pickle.load(f)
        if variant["traj_num"] > 0:
            traj_list = random.sample(traj_list, variant["traj_num"])

        obs = np.vstack(
            [
                traj_list[i][k]["observations"]
                for i in range(len(traj_list))
                for k in traj_list[i].keys()
            ]
        )
        obs_mean, obs_std = np.mean(obs, axis=0), np.std(obs, axis=0)
        print("mean:{}\nstd:{}".format(obs_mean, obs_std))

        _env_wrapper = env_wrapper
        env_wrapper = lambda *args, **kwargs: ObsScaledEnv(
            _env_wrapper(*args, **kwargs),
            obs_mean=obs_mean,
            obs_std=obs_std,
        )

    env = env_wrapper(env)

    policy = joblib.load(variant["policy_checkpoint"])["policy_0"]["policy"]

    if variant["eval_deterministic"]:
        policy = MakeDeterministic(policy)
    policy.to(ptu.device)

    for _ in range(variant["num_vehicles"]):
        observation_n = env.reset()
        for step in range(variant["max_path_length"]):
            action_n = {}
            for agent_id, observation in observation_n.items():
                action, _ = policy.get_action(obs_np=observation)
                action_n[agent_id] = action

            next_observation_n, reward_n, terminal_n, env_info_n = env.step(action_n)

            for agent_id in terminal_n.keys():
                if terminal_n[agent_id]:
                    car_id = env_info_n[agent_id]["car_id"]
                    print(f"car {car_id} terminated @ {step}")
            observation_n = next_observation_n


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", help="experiment specification file")
    parser.add_argument("-g", "--gpu", help="gpu id", type=int, default=0)

    args = parser.parse_args()
    with open(args.experiment, "r") as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string, Loader=yaml.Loader)

    if exp_specs["using_gpus"]:
        print("\nUSING GPU\n")
        ptu.set_gpu_mode(True)

    exp_id = exp_specs["exp_id"]
    exp_prefix = exp_specs["exp_name"]

    pkl_name = "params.pkl"
    exp_specs["policy_checkpoint"] = os.path.join(exp_specs["log_path"], pkl_name)
    exp_specs["video_path"] = os.path.join(exp_specs["log_path"], "videos")
    if not os.path.exists(exp_specs["video_path"]):
        os.mkdir(exp_specs["video_path"])

    envision_proc = Popen(
        f"scl envision start -s {ScenarioZoo.get_scenario('NGSIM-I80')} -p 8081", shell=True
    )

    seed = exp_specs["seed"]
    set_seed(seed)
    try:
        experiment(exp_specs)
    except Exception as e:
        os.killpg(os.getpgid(envision_proc.pid), signal.SIGKILL)
        raise e

    os.killpg(os.getpgid(envision_proc.pid), signal.SIGKILL)
    envision_proc.wait()
