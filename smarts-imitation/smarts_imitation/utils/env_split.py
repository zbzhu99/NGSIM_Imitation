import numpy as np
import math
from collections import OrderedDict


def split_vehicles(scenario_vehicles, env_num):
    """
    Args:
        scenario_vehicles (OrderedDict): {traffic_name: [vehicles,...], }
        env_num (int): env num to split.
    Returns:
        splitted_vehicles (OrderedDict): {traffic_name: [[vehicles], ] ,}
        real_env_num: real env num to be splitted to.
    Returns
    """
    splitted_vehicles = OrderedDict()
    total_vehicle_num = sum([len(x) for x in scenario_vehicles.values()])
    real_env_num = 0
    for (
        traffic_name,
        traffic_vehicles,
    ) in scenario_vehicles.items():  # keep traffic name to be ordered.
        traffic_env_num = int(env_num * len(traffic_vehicles) / total_vehicle_num + 0.5)
        vehicles_lists = [[] for _ in range(traffic_env_num)]
        for i in range(len(traffic_vehicles)):
            vehicles_lists[i % traffic_env_num].append(traffic_vehicles[i])
        real_env_num += traffic_env_num
        splitted_vehicles[traffic_name] = vehicles_lists

    return splitted_vehicles, real_env_num
