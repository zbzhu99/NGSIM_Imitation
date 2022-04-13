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
    env_specs,
    env_wrapper=None,
    vehicle_ids_list_mapping: dict = {},
    env_num=1,
    wait_num=None,
    auto_reset=False,
    seed=None,
    **kwargs,
):
    """
    env_specs:
        env_name: 'mujoco'
        scenario_name: 'halfcheetah'
        env_kwargs: {} # kwargs to pass to the env constructor call
        vehicle_ids_list: {traffic_name: [[vehicle_ids],...]}
    """
    assert env_num == sum([len(x) for x in vehicle_ids_list_mapping.values()])
    if env_wrapper is None:
        env_wrapper = ProxyEnv

    try:
        env_class = get_env_cls(env_specs["env_creator"])
    except KeyError:
        print("Unknown env name: {}".format(env_specs["env_creator"]))

    env_fns = []
    for traffic_name, traffic_vehicle_ids_lists in vehicle_ids_list_mapping.items():
        for traffic_vehicle_ids_list in traffic_vehicle_ids_lists:
            env_fns.append(
                lambda ids=traffic_vehicle_ids_list, name=traffic_name: env_wrapper(
                    env_class(traffic_name=name, vehicle_ids=ids, **env_specs)
                )
            )
    envs = SubprocVectorEnv(
        env_fns,
        wait_num=wait_num,
        auto_reset=auto_reset,
        **kwargs,
    )

    envs.seed(seed)
    return envs
