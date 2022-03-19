import os
import sys
import yaml
import inspect
import argparse

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
print(sys.path)

import gym
from rlkit.envs import get_env, get_envs
from rlkit.envs.wrappers import NormalizedBoxActEnv, ProxyEnv

import rlkit.torch.utils.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger, set_seed
from rlkit.torch.common.networks import FlattenMlp
from rlkit.torch.common.policies import DiscretePolicy
from rlkit.torch.algorithms.discrete_sac.discrete_sac import DiscreteSoftActorCritic
from rlkit.torch.algorithms.torch_rl_algorithm import TorchRLAlgorithm


def experiment(variant):
    env_specs = variant["env_specs"]
    env = get_env(env_specs)
    env.seed(env_specs["eval_env_seed"])

    print("\n\nEnv: {}:{}".format(env_specs["env_creator"], env_specs["env_name"]))
    print("kwargs: {}".format(env_specs["env_kwargs"]))
    print("Obs Space: {}".format(env.observation_space_n))
    print("Act Space: {}\n\n".format(env.action_space_n))

    obs_space_n = env.observation_space_n
    act_space_n = env.action_space_n

    policy_mapping_dict = dict(
        zip(env.agent_ids, ["policy_0" for _ in range(env.n_agents)])
    )

    policy_trainer_n = {}
    policy_n = {}

    # create policies
    for agent_id in env.agent_ids:
        policy_id = policy_mapping_dict.get(agent_id)
        if policy_id not in policy_trainer_n:
            print(f"Create {policy_id} for {agent_id} ...")
            obs_space = obs_space_n[agent_id]
            act_space = act_space_n[agent_id]
            assert isinstance(obs_space, gym.spaces.Box)
            assert isinstance(act_space, gym.spaces.Discrete)
            assert len(obs_space.shape) == 1

            obs_dim = obs_space_n[agent_id].shape[0]
            action_dim = act_space_n[agent_id].n

            net_size = variant["net_size"]
            num_hidden = variant["num_hidden_layers"]
            qf1 = FlattenMlp(
                hidden_sizes=num_hidden * [net_size],
                input_size=obs_dim,
                output_size=action_dim,
            )
            qf2 = FlattenMlp(
                hidden_sizes=num_hidden * [net_size],
                input_size=obs_dim,
                output_size=action_dim,
            )
            policy = DiscretePolicy(
                hidden_sizes=num_hidden * [net_size],
                obs_dim=obs_dim,
                action_dim=action_dim,
            )

            trainer = DiscreteSoftActorCritic(
                policy=policy, qf1=qf1, qf2=qf2, **variant["sac_params"]
            )
            policy_trainer_n[policy_id] = trainer
            policy_n[policy_id] = policy
        else:
            print(f"Use existing {policy_id} for {agent_id} ...")

    env_wrapper = ProxyEnv  # Identical wrapper

    env = env_wrapper(env)
    training_env = get_envs(env_specs, env_wrapper)
    training_env.seed(env_specs["training_env_seed"])

    algorithm = TorchRLAlgorithm(
        trainer_n=policy_trainer_n,
        env=env,
        training_env=training_env,
        exploration_policy_n=policy_n,
        policy_mapping_dict=policy_mapping_dict,
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

    # make all seeds the same.
    exp_specs["env_specs"]["eval_env_seed"] = exp_specs["env_specs"][
        "training_env_seed"
    ] = exp_specs["seed"]

    if exp_specs["using_gpus"] > 0:
        print("\n\nUSING GPU\n\n")
        ptu.set_gpu_mode(True, args.gpu)
    exp_id = exp_specs["exp_id"]
    exp_prefix = exp_specs["exp_name"]
    seed = exp_specs["seed"]
    set_seed(seed)
    setup_logger(exp_prefix=exp_prefix, exp_id=exp_id, variant=exp_specs)

    experiment(exp_specs)
