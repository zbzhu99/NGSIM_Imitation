from collections import OrderedDict, defaultdict
from functools import partial


def to_ordered_dict(origin_dict):
    if not isinstance(origin_dict, dict):
        return origin_dict
    ordered_dict = OrderedDict()
    for key, value in sorted(origin_dict.items()):
        ordered_dict[key] = to_ordered_dict(value)
    return ordered_dict


def split_vehicles(scenario_vehicles, env_num):
    """
    Args:
        scenario_vehicles (OrderedDict): {scenario_name: {traffic_name: [vehicles,...], }}
        env_num (int): env num to split.
    Returns:
        splitted_vehicles (OrderedDict): {traffic_name: [[vehicles], ] ,}
        real_env_num: real env num to be splitted to.
    Returns
    """
    # splitted_vehicles = OrderedDict()
    splitted_vehicles = defaultdict(partial(defaultdict, list))
    total_vehicle_num = sum(
        [
            sum([len(vehicles) for vehicles in traffics.values()])
            for traffics in scenario_vehicles.values()
        ]
    )
    real_env_num = 0
    for scenario_name, traffics in scenario_vehicles.items():
        for traffic_name, vehicles in traffics.items():
            traffic_env_num = int(env_num * len(vehicles) / total_vehicle_num + 0.5)
            vehicles_lists = [[] for _ in range(traffic_env_num)]
            for i in range(len(vehicles)):
                vehicles_lists[i % traffic_env_num].append(vehicles[i])
            real_env_num += traffic_env_num
            splitted_vehicles[scenario_name][traffic_name] = vehicles_lists

    return splitted_vehicles, real_env_num
