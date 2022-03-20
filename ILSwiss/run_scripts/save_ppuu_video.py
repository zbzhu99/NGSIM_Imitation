import yaml
import argparse
import joblib
import os
import sys
import inspect
import pickle
import numpy as np

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
print(sys.path)

from rlkit.envs import get_env

import rlkit.torch.utils.pytorch_util as ptu
from rlkit.launchers.launcher_util import set_seed

from rlkit.torch.common.policies import MakeDeterministic

from video import save_video


def experiment(variant):
    with open("demos_listing.yaml", "r") as f:
        listings = yaml.load(f.read(), Loader=yaml.FullLoader)

    eval_split_path = listings[variant["expert_name"]]["train_split"][0]
    with open(eval_split_path, "rb") as f:
        eval_vehicle_ids = pickle.load(f)

    eval_vehicle_ids = np.array(eval_vehicle_ids)
    # Can specify vehicle ids to be visualized as follows.
    # eval_vehicle_ids = np.array([[0, 788],])
    variant["num_vehicles"] = len(eval_vehicle_ids)
    print("Total Vehicle Size: ", len(eval_vehicle_ids))

    env_specs = variant["env_specs"]
    env = get_env(env_specs, vehicle_ids=eval_vehicle_ids)
    env.seed(variant["seed"])

    print("\nEnv: {}: {}".format(env_specs["env_creator"], env_specs["scenario_name"]))
    print("kwargs: {}".format(env_specs["env_kwargs"]))
    print("Obs Space: {}".format(env.observation_space_n))
    print("Act Space: {}\n".format(env.action_space_n))

    if variant["scale_env_with_demo_stats"]:
        # The normalization is implemented inside the PPUU simulator.
        raise NotImplementedError

    agent_id = "agent_0"
    policy = joblib.load(variant["policy_checkpoint"])["policy_0"]["policy"]

    if variant["eval_deterministic"]:
        policy = MakeDeterministic(policy)
    policy.to(ptu.device)

    fps = variant["fps"]
    video_path = variant["video_path"]
    for _ in range(variant["num_vehicles"]):
        images = []
        observation_n = env.reset()

        image, _ = env.get_render_image()
        observation = observation_n[agent_id]
        car_id = None

        for step in range(variant["max_path_length"]):
            action, agent_info = policy.get_action(obs_np=observation)

            action_n = {agent_id: action}
            next_observation_n, reward_n, terminal_n, env_info_n = env.step(action_n)
            if car_id is None:
                car_id = env_info_n[agent_id]["car_id"]

            image, _ = env.get_render_image()
            images.append(image)

            if terminal_n[agent_id]:
                print(f"car {car_id} terminated @ {step}")
                break
            observation = next_observation_n[agent_id]

        print(f"saving videos of car {car_id}...")
        images = np.stack(images, axis=0)
        video_save_path = os.path.join(video_path, f"car_{car_id}.mp4")
        save_video(images, video_save_path, fps=fps)


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

    seed = exp_specs["seed"]
    set_seed(seed)
    experiment(exp_specs)
