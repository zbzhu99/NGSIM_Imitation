import numpy as np
from collections import OrderedDict


def split_vehicle_ids(vehicle_ids, env_num):
    """
    Args:
        vehicle_ids (OrderedDict): {traffic_name: [vehicle_ids,...], }
        env_num (int): env num to split.
    Returns:
        splitted_vehicle_ids (OrderedDict): {traffic_name: [[vehicle_ids], ] ,}
        real_env_num: real env num to be splitted to.
    Returns
    """
    splitted_vehicle_ids = OrderedDict()
    total_vehicle_num = sum([len(x) for x in vehicle_ids.values()])
    real_env_num = 0
    for (
        traffic_name,
        traffic_vehicle_ids,
    ) in vehicle_ids.items():  # keep traffic name to be ordered.
        traffic_env_num = int(
            env_num * len(traffic_vehicle_ids) / total_vehicle_num + 0.5
        )
        real_env_num += traffic_env_num
        splitted_vehicle_ids[traffic_name] = np.array_split(
            traffic_vehicle_ids, traffic_env_num
        )
    return splitted_vehicle_ids, real_env_num
