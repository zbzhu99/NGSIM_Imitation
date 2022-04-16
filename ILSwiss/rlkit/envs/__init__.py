from typing import Dict, Callable, Any
from gym import Env
import numpy as np

from rlkit.env_creators import get_env_cls
from rlkit.envs.wrappers import ProxyEnv
from rlkit.envs.vecenvs import BaseVectorEnv, DummyVectorEnv, SubprocVectorEnv

__all__ = [
    "BaseVectorEnv",
    "DummyVectorEnv",
    "SubprocVectorEnv",
]


def get_env(env_specs, traffic_name, vehicle_ids=None):
    """
    env_specs:
        env_name: 'mujoco'
        scenario_name: 'halfcheetah'
        env_kwargs: {} # kwargs to pass to the env constructor call
    """
    try:
        env_class = get_env_cls(env_specs["env_creator"])
    except KeyError:
        print("Unknown env name: {}".format(env_specs["env_creator"]))

    env = env_class(traffic_name=traffic_name, vehicle_ids=vehicle_ids, **env_specs)

    return env


def get_envs(
    env_specs: Dict[str, Any],
    env_wrapper: Callable[..., Env] = None,
    splitted_vehicle_ids: Dict[str, np.ndarray] = {},
    env_num: int = 1,
    wait_num: int = None,
    auto_reset: bool = False,
    seed: int = None,
    **kwargs,
):
    """
    env_specs:
        env_name: 'mujoco'
        scenario_name: 'halfcheetah'
        env_kwargs: {} # kwargs to pass to the env constructor call
        splitted_vehicle_ids_list: {traffic_name: [[vehicle_ids],...]}
    """

    assert env_num == sum([len(x) for x in splitted_vehicle_ids.values()])
    if env_wrapper is None:
        env_wrapper = ProxyEnv

    try:
        env_class = get_env_cls(env_specs["env_creator"])
    except KeyError:
        print("Unknown env name: {}".format(env_specs["env_creator"]))

    env_fns = []
    for traffic_name, traffic_vehicle_ids_lists in splitted_vehicle_ids.items():
        for traffic_vehicle_ids_list in traffic_vehicle_ids_lists:
            env_fns.append(
                lambda ids=traffic_vehicle_ids_list, name=traffic_name: env_wrapper(
                    env_class(traffic_name=name, vehicle_ids=ids, **env_specs)
                )
            )

    if env_num == 1:
        print("\n WARNING: Single environment detected, wrap to DummyVectorEnv.\n")
        # Dummy vector env is kept for debugging purpose.
        envs = DummyVectorEnv(env_fns, auto_reset=auto_reset, **kwargs)
    else:
        envs = SubprocVectorEnv(
            env_fns,
            wait_num=wait_num,
            auto_reset=auto_reset,
            **kwargs,
        )

    envs.seed(seed)
    return envs
