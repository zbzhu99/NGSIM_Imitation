import joblib
import torch
import os
import sys
import inspect
import pickle
import numpy as np
import json

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
print(sys.path)

configs = {
    "policy_input_path": "/home/liuyuecheng/YuechengLiu-NGSIM-Imitation/0505/ILSwiss/logs/gail-smarts-all-scenarios-cutin--scale--terminal--gp-4.0--rs-4.0--trajnum--1--stack-1/gail_smarts_all_scenarios_cutin--scale--terminal--gp-4.0--rs-4.0--trajnum--1--stack-1_2022_05_11_10_08_21_0000--s-0/params.pkl",
    "polity_output_path": "./cpp_model/cpp_model.ckpt",
    "stats_input_path": "/home/liuyuecheng/YuechengLiu-NGSIM-Imitation/0505/ILSwiss/demos/ngsim_i80_ngsim_us101/smarts_radius_stack-1_cutin.pkl",
    "stats_output_path": "./cpp_model/stats.json",
}


def save_policy():
    input_path = configs["policy_input_path"]
    output_path = configs["polity_output_path"]
    output_dir_name = os.path.dirname(output_path)
    if not os.path.exists(output_dir_name):
        os.makedirs(output_dir_name)

    policy = joblib.load(input_path)["policy_0"]["policy"]
    policy.to("cpu")
    policy_script_model = torch.jit.script(policy)
    print(f"ScriptPolicyModel: {policy_script_model}")
    policy_script_model.save(output_path)
    print("save policy finished!")


def save_stats():
    demos_path = configs["stats_input_path"]
    output_path = configs["stats_output_path"]
    output_dir_name = os.path.dirname(output_path)
    if not os.path.exists(output_dir_name):
        os.makedirs(output_dir_name)

    with open(demos_path, "rb") as f:
        traj_list = pickle.load(f)
    obs = np.vstack(
        [
            traj_list[i][k]["observations"]
            for i in range(len(traj_list))
            for k in traj_list[i].keys()
        ]
    )

    obs_mean, obs_std = np.mean(obs, axis=0), np.std(obs, axis=0)
    mean = obs_mean.tolist()
    std = obs_std.tolist()
    stats = {
        "mean": mean,
        "std": std,
    }
    shape = {key: len(value) for key, value in stats.items()}
    with open(output_path, "w") as f:
        json.dump(stats, f)
    print(f"stats: {stats} \nstats shape: {shape}")
    print("save stats finished!")


def main():

    save_policy()
    save_stats()


if __name__ == "__main__":
    main()
