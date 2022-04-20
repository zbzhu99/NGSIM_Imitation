import yaml
import argparse
import os
import sys
import inspect
import pickle

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
print(sys.path)

import gym
from rlkit.envs import get_env, get_envs
from rlkit.envs.wrappers import NormalizedBoxEnv, ProxyEnv

import rlkit.torch.utils.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger, set_seed
from rlkit.torch.common.networks import FlattenMlp
from rlkit.torch.common.policies import ReparamTanhMultivariateGaussianPolicy
from rlkit.torch.algorithms.sac.sac_alpha import SoftActorCritic
from rlkit.torch.algorithms.torch_rl_algorithm import TorchRLAlgorithm
from smarts_imitation.utils.env_split import split_vehicles


def experiment(variant):
    with open("demos_listing.yaml", "r") as f:
        listings = yaml.load(f.read(), Loader=yaml.FullLoader)

    train_split_path = listings[variant["expert_name"]]["train_split"][0]
    with open(train_split_path, "rb") as f:
        # train_vehicle_ids is a OrderedDcit
        train_vehicles = pickle.load(f)

    env_specs = variant["env_specs"]
    env = get_env(env_specs, traffic_name=list(train_vehicles.keys())[0])
    env.seed(env_specs["eval_env_seed"])

    print("\n\nEnv: {}".format(env_specs["env_name"]))
    print("kwargs: {}".format(env_specs["env_kwargs"]))
    print("Obs Space: {}".format(env.observation_space))
    print("Act Space: {}\n\n".format(env.action_space))

    obs_space = env.observation_space
    act_space = env.action_space
    assert isinstance(obs_space, gym.spaces.Box)
    assert isinstance(act_space, gym.spaces.Box)
    assert len(obs_space.shape) == 1
    assert len(act_space.shape) == 1

    env_wrapper = ProxyEnv  # Identical wrapper
    if isinstance(act_space, gym.spaces.Box):
        env_wrapper = NormalizedBoxEnv
        kwargs = {}

    env = env_wrapper(env, **kwargs)
    train_splitted_vehicles, train_real_env_num = split_vehicles(
        train_vehicles, env_specs["training_env_specs"]["env_num"]
    )
    train_env_nums = {
        traffic_name: len(vehicles_list)
        for traffic_name, vehicles_list in train_splitted_vehicles.items()
    }
    print("training env nums: {}".format(train_env_nums))
    env_specs["training_env_specs"]["env_num"] = train_real_env_num
    env_specs["training_env_specs"]["wait_num"] = min(
        train_real_env_num, env_specs["training_env_specs"]["wait_num"]
    )
    training_env = get_envs(
        env_specs,
        env_wrapper,
        splitted_vehicles=train_splitted_vehicles,
        **kwargs,
    )
    training_env.seed(env_specs["training_env_seed"])

    obs_dim = obs_space.shape[0]
    action_dim = act_space.shape[0]

    net_size = variant["net_size"]
    num_hidden = variant["num_hidden_layers"]
    qf1 = FlattenMlp(
        hidden_sizes=num_hidden * [net_size],
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    qf2 = FlattenMlp(
        hidden_sizes=num_hidden * [net_size],
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    vf = FlattenMlp(
        hidden_sizes=num_hidden * [net_size],
        input_size=obs_dim,
        output_size=1,
    )
    policy = ReparamTanhMultivariateGaussianPolicy(
        hidden_sizes=num_hidden * [net_size],
        obs_dim=obs_dim,
        action_dim=action_dim,
    )

    trainer = SoftActorCritic(
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        vf=vf,
        **variant["sac_params"],
    )
    algorithm = TorchRLAlgorithm(
        trainer=trainer,
        env=env,
        training_env=training_env,
        exploration_policy=policy,
        **variant["rl_alg_params"],
    )

    if ptu.gpu_enabled():
        algorithm.to(ptu.device)
    algorithm.train()

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

    if exp_specs["num_gpu_per_worker"] > 0:
        print("\n\nUSING GPU\n\n")
        ptu.set_gpu_mode(True, args.gpu)
    exp_id = exp_specs["exp_id"]
    exp_prefix = exp_specs["exp_name"]
    seed = exp_specs["seed"]
    set_seed(seed)
    setup_logger(exp_prefix=exp_prefix, exp_id=exp_id, variant=exp_specs)

    experiment(exp_specs)
