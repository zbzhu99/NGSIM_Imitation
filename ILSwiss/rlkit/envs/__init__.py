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


def get_env(env_specs, scenario_name, traffic_name, vehicles=None):
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

    env = env_class(
        scenario_name=scenario_name,
        traffic_name=traffic_name,
        vehicles=vehicles,
        **env_specs,
    )

    return env


def get_envs(
    env_specs: Dict[str, Any],
    env_wrapper: Callable[..., Env] = None,
    splitted_vehicles: Dict[str, Dict[str, list]] = {},
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
        splitted_vehicle_s: {scenario_name: {traffic_name: [[vehicle_ids],...]}, }
    """

    assert env_num == sum(
        [
            sum([len(x) for x in traffics.values()])
            for traffics in splitted_vehicles.values()
        ]
    )
    if env_wrapper is None:
        env_wrapper = ProxyEnv

    try:
        env_class = get_env_cls(env_specs["env_creator"])
    except KeyError:
        print("Unknown env name: {}".format(env_specs["env_creator"]))

    env_fns = []
    for scenario_name, traffics in splitted_vehicles.items():
        for traffic_name, vehicles in traffics.items():
            env_fns.append(
                lambda s_name=scenario_name, t_name=traffic_name, vs=vehicles: env_wrapper(
                    env_class(
                        scenario_name=s_name,
                        traffic_name=t_name,
                        vehicles=vs,
                        **env_specs,
                    )
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
