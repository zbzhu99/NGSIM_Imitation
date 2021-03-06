import yaml
import argparse
import joblib
import numpy as np
import os
import sys
import inspect
import pickle
import random
import gym

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
print(sys.path)

from rlkit.envs import get_env, get_envs
import rlkit.torch.utils.pytorch_util as ptu
from rlkit.launchers.launcher_util import set_seed
from rlkit.core import eval_util
from rlkit.envs.wrappers import ProxyEnv, NormalizedBoxActEnv, ObsScaledEnv
from rlkit.samplers import PathSampler
from rlkit.torch.common.policies import MakeDeterministic
from smarts_imitation.utils.env_split import split_vehicles


def experiment(variant):
    env_specs = variant["env_specs"]

    with open("demos_listing.yaml", "r") as f:
        listings = yaml.load(f.read(), Loader=yaml.FullLoader)
    eval_split_path = listings[variant["expert_name"]]["eval_split"][0]
    with open(eval_split_path, "rb") as f:
        # eval_vehicle_ids is a OrderedDict
        eval_vehicles = pickle.load(f)

    """ 1. Create Template Env and Eval Vector Envs """
    s_name = list(eval_vehicles.keys())[0]
    t_name = list(eval_vehicles[s_name]).keys()[0]
    env = get_env(env_specs, scenario_name=s_name, traffic_name=t_name)
    env.seed(env_specs["eval_env_seed"])

    print("\nEnv: {}".format(env_specs["env_creator"]))
    print("kwargs: {}".format(env_specs["env_kwargs"]))
    print("Obs Space: {}".format(env.observation_space_n))
    print("Act Space: {}\n".format(env.action_space_n))

    env_wrapper = ProxyEnv  # Identical wrapper
    for act_space in env.action_space_n.values():
        if isinstance(act_space, gym.spaces.Box):
            env_wrapper = NormalizedBoxActEnv
            break

    if variant["scale_env_with_demo_stats"]:
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

    eval_splitted_vehicles, eval_real_env_num = split_vehicles(
        eval_vehicles, env_specs["eval_env_specs"]["env_num"]
    )
    eval_env_nums = {
        scenario_name: {
            traffic_name: len(vehicles) for traffic_name, vehicles in traffics.items()
        }
        for scenario_name, traffics in eval_splitted_vehicles.items()
    }
    print("eval env nums: {}".format(eval_env_nums))
    env_specs["eval_env_specs"]["env_num"] = eval_real_env_num
    env_specs["eval_env_specs"]["wait_num"] = min(
        eval_real_env_num, env_specs["eval_env_specs"]["wait_num"]
    )

    # Vehicles are splitted and assigned to different vector environments, and those from
    # different envs will not be controlled together. Thus, evaluation result will vary with
    # the number of evaluation envs when env.control_vehicle_num > 1.

    eval_env = get_envs(
        env_specs,
        env_wrapper,
        splitted_vehicles=eval_splitted_vehicles,
        **env_specs["eval_env_specs"],
    )
    eval_car_num = []
    for traffics in eval_splitted_vehicles.values():
        # should be ordered.
        for vehicles_lists in traffics.values():
            eval_car_num.extend(
                [len(x) - env.control_vehicle_num + 1 for x in vehicles_lists]
            )
    eval_car_num = [
        eval_env.sub_envs_info[env_id].vehicle_num - env.control_vehicle_num + 1
        for env_id in range(eval_real_env_num)
    ]

    """ 2. Load Checkpoint Policies """
    # all agents share the same policy
    policy_mapping_dict = dict(
        zip(env.agent_ids, ["policy_0" for _ in range(env.n_agents)])
    )

    policy_n = {}
    for policy_id in policy_mapping_dict.values():
        policy = joblib.load(variant["policy_checkpoint"])[policy_id]["policy"]
        if variant["eval_deterministic"]:
            policy = MakeDeterministic(policy)
        policy.to(ptu.device)
        policy_n[policy_id] = policy

    """ 3. Sample Trajectories """
    eval_sampler = PathSampler(
        env,
        eval_env,
        policy_n,
        policy_mapping_dict,
        variant["num_eval_steps"],
        variant["max_path_length"],
        car_num=eval_car_num,
        no_terminal=variant["no_terminal"],
        render=variant["render"],
        render_kwargs=variant["render_kwargs"],
    )
    test_paths = eval_sampler.obtain_samples()

    """ 4. Compute Statistics """
    statistics = eval_util.get_generic_path_information(
        test_paths,
        env,
        stat_prefix="Test",
    )
    return statistics, test_paths


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", help="experiment specification file")
    parser.add_argument("-g", "--gpu", help="gpu id", type=int, default=0)
    parser.add_argument(
        "-s", "--save_res", help="save result to file", type=int, default=1
    )

    args = parser.parse_args()
    with open(args.experiment, "r") as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string, Loader=yaml.FullLoader)

    # make all seeds the same.
    exp_specs["env_specs"]["eval_env_seed"] = exp_specs["seed"]

    if exp_specs["using_gpus"] > 0:
        print("\nUSING GPU\n")
        ptu.set_gpu_mode(True)
    exp_id = exp_specs["exp_id"]
    exp_prefix = exp_specs["exp_name"]

    seed = exp_specs["seed"]
    set_seed(seed)

    exp_specs["policy_checkpoint"] = os.path.join(exp_specs["log_path"], "best.pkl")
    statistics, test_paths = experiment(exp_specs)

    if args.save_res:
        save_path = os.path.join(exp_specs["log_path"], "res.csv")

        if not os.path.exists(save_path):
            with open(save_path, "w") as f:
                f.write("success_rate,avg_distance,std_distance\n")
        with open(save_path, "a") as f:
            f.write(
                "{},{},{}\n".format(
                    statistics["Test agent_0 Success Rate"],
                    statistics["Test agent_0 Distance Mean"],
                    statistics["Test agent_0 Distance Std"],
                )
            )

    if exp_specs["save_samples"]:
        with open(
            os.path.join(exp_specs["log_path"], "samples.pkl"),
            "wb",
        ) as f:
            pickle.dump(test_paths, f)
