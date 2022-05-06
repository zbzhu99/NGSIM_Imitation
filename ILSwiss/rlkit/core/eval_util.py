"""
Common evaluation utilities.
"""

from collections import OrderedDict
from numbers import Number
import os
import json

import numpy as np

from rlkit.core.vistools import plot_returns_on_same_plot
from collections import defaultdict
from functools import partial


class ScenarioStats:
    _attr_mapping = {
        "success_rate": "win_count",
        "collision_rate": "collision_count",
        "dist_to_hist_cur_pos_sum": "dist_to_hist_cur_pos_sum",
        "dist_to_hist_cur_pos_mean": "dist_to_hist_cur_pos_mean",
        "dist_to_hist_final_pos": "dist_to_hist_final_pos",
        "lane_change": "lane_change",
    }

    def __init__(self, agent_ids):
        self.agent_ids = agent_ids

        self.scenario_total_count = defaultdict(partial(defaultdict, float))
        self.all_scenarios_total_count = defaultdict(float)
        for attr_name in ScenarioStats._attr_mapping.values():
            setattr(
                self,
                "all_scenarios_" + attr_name,
                defaultdict(float),
            )
            setattr(
                self, "scenario_" + attr_name, defaultdict(partial(defaultdict, float))
            )

    def reset(self):
        self.scenario_total_count = defaultdict(partial(defaultdict, float))
        self.all_scenarios_total_count = defaultdict(float)
        for attr_name in ScenarioStats._attr_mapping.values():
            setattr(
                self,
                "all_scenarios_" + attr_name,
                defaultdict(partial(defaultdict, float)),
            )
            setattr(
                self, "scenario_" + attr_name, defaultdict(partial(defaultdict, float))
            )

    def update(self, paths):
        for a_id in self.agent_ids:
            for path in paths:
                scenario_name = path.scenario_name

                data = {
                    "success_rate": None,
                    "collision_rate": None,
                    "dist_to_hist_cur_pos_sum": None,
                    "dist_to_hist_cur_pos_mean": None,
                    "dist_to_hist_final_pos": None,
                    "lane_change": None,
                }

                data["success_rate"] = float(
                    path[a_id]["env_infos"][-1]["reached_goal"]
                )
                data["collision_rate"] = float(path[a_id]["env_infos"][-1]["collision"])
                dist_to_hist_cur_pos = []
                for i in range(len(path[a_id]["env_infos"])):
                    dist = path[a_id]["env_infos"][i]["dist_to_hist_cur_pos"]
                    if dist is None:
                        continue
                    dist_to_hist_cur_pos.append(dist)

                if len(dist_to_hist_cur_pos) > 0:
                    dist_to_hist_cur_pos_sum = sum(dist_to_hist_cur_pos)
                    dist_to_hist_cur_pos_mean = dist_to_hist_cur_pos_sum / len(
                        dist_to_hist_cur_pos
                    )
                    data["dist_to_hist_cur_pos_sum"] = dist_to_hist_cur_pos_sum
                    data["dist_to_hist_cur_pos_mean"] = dist_to_hist_cur_pos_mean

                data["dist_to_hist_final_pos"] = path[a_id]["env_infos"][-1][
                    "dist_to_hist_cur_pos"
                ]

                lane_change = 0
                for i in range(1, len(path[a_id]["env_infos"])):
                    pre_lane = path[a_id]["env_infos"][i - 1]["lane_index"]
                    cur_lane = path[a_id]["env_infos"][i]["lane_index"]
                    lane_change += float(cur_lane != pre_lane)
                data["lane_change"] = lane_change

                for name, value in data.items():
                    if value is None:
                        continue
                    all_attr = getattr(
                        self, "all_scenarios_" + ScenarioStats._attr_mapping[name]
                    )
                    scenario_attr = getattr(
                        self, "scenario_" + ScenarioStats._attr_mapping[name]
                    )
                    all_attr[a_id] += value
                    scenario_attr[a_id][scenario_name] += value

                self.scenario_total_count[a_id][scenario_name] += 1
                self.all_scenarios_total_count[a_id] += 1

    def get_stats(self, name):

        assert name in ScenarioStats._attr_mapping
        stats = defaultdict(partial(defaultdict, float))
        for a_id, scenarios in self.scenario_total_count.items():
            for scenario_name, s_total_count in scenarios.items():
                if s_total_count == 0:
                    pass
                else:
                    _scenario_value = getattr(
                        self, "scenario_" + ScenarioStats._attr_mapping[name]
                    )[a_id][scenario_name]
                    stats[a_id][scenario_name] = _scenario_value / s_total_count
            all_total_count = self.all_scenarios_total_count[a_id]
            if all_total_count == 0:
                pass
            else:
                all_value = getattr(
                    self, "all_scenarios_" + ScenarioStats._attr_mapping[name]
                )[a_id]
                stats[a_id]["all"] = all_value / all_total_count
        return stats


class InfoAdvIRLScenarioStats(ScenarioStats):
    def update(self, paths):
        for a_id in self.agent_ids:
            for path in paths:
                scenario_name = path.scenario_name
                latent_id = "latent_id_{}".format(path[a_id]["latents"][0])

                data = {
                    "success_rate": None,
                    "collision_rate": None,
                    "dist_to_hist_cur_pos_sum": None,
                    "dist_to_hist_cur_pos_mean": None,
                    "dist_to_hist_final_pos": None,
                    "lane_change": None,
                }

                data["success_rate"] = float(
                    path[a_id]["env_infos"][-1]["reached_goal"]
                )
                data["collision_rate"] = float(path[a_id]["env_infos"][-1]["collision"])
                dist_to_hist_cur_pos = []
                for i in range(len(path[a_id]["env_infos"])):
                    dist = path[a_id]["env_infos"][i]["dist_to_hist_cur_pos"]
                    if dist is None:
                        continue
                    dist_to_hist_cur_pos.append(dist)

                if len(dist_to_hist_cur_pos) > 0:
                    dist_to_hist_cur_pos_sum = sum(dist_to_hist_cur_pos)
                    dist_to_hist_cur_pos_mean = dist_to_hist_cur_pos_sum / len(
                        dist_to_hist_cur_pos
                    )
                    data["dist_to_hist_cur_pos_sum"] = dist_to_hist_cur_pos_sum
                    data["dist_to_hist_cur_pos_mean"] = dist_to_hist_cur_pos_mean

                data["dist_to_hist_final_pos"] = path[a_id]["env_infos"][-1][
                    "dist_to_hist_cur_pos"
                ]

                lane_change = 0
                for i in range(1, len(path[a_id]["env_infos"])):
                    pre_lane = path[a_id]["env_infos"][i - 1]["lane_index"]
                    cur_lane = path[a_id]["env_infos"][i]["lane_index"]
                    lane_change += float(cur_lane != pre_lane)
                data["lane_change"] = lane_change

                for name, value in data.items():
                    if value is None:
                        continue
                    all_attr = getattr(
                        self, "all_scenarios_" + ScenarioStats._attr_mapping[name]
                    )
                    scenario_attr = getattr(
                        self, "scenario_" + ScenarioStats._attr_mapping[name]
                    )
                    all_attr[a_id] += value
                    scenario_attr[a_id][scenario_name] += value
                    scenario_attr[a_id][latent_id] += value

                self.scenario_total_count[a_id][scenario_name] += 1
                self.scenario_total_count[a_id][latent_id] += 1
                self.all_scenarios_total_count[a_id] += 1


def get_generic_path_information(
    paths, env, stat_prefix="", scenario_stats_class=ScenarioStats
):
    """
    Get an OrderedDict with a bunch of statistic names and values.
    """
    # XXX(zbzhu): maybe consider a better way to get `agent_ids`
    agent_ids = paths[0].agent_ids

    """ Bunch of deprecated codes and comments. Will be removed soon.
    # returns = [sum(path["rewards"]) for path in paths]
    # rewards = np.vstack([path["rewards"] for path in paths])
    # rewards = np.concatenate([path["rewards"] for path in paths])
    # if isinstance(actions[0][0], np.ndarray):
    #     actions = np.vstack([path["actions"] for path in paths])
    # else:
    #     actions = np.hstack([path["actions"] for path in paths])
    """

    statistics = OrderedDict()

    distance_travelled_n = {}
    for a_id in agent_ids:
        distance_travelled = []
        for path in paths:
            distance = 0
            for i in range(1, len(path[a_id]["env_infos"])):
                distance += np.linalg.norm(
                    path[a_id]["env_infos"][i]["raw_position"][:2]
                    - path[a_id]["env_infos"][i - 1]["raw_position"][:2]
                )
            distance_travelled.append(distance)
        distance_travelled_n[a_id] = distance_travelled

    scenario_stats = scenario_stats_class(agent_ids)
    scenario_stats.update(paths)
    success_rate_n = scenario_stats.get_stats("success_rate")
    collision_rate_n = scenario_stats.get_stats("collision_rate")
    dist_to_hist_cur_pos_sum_n = scenario_stats.get_stats("dist_to_hist_cur_pos_sum")
    dist_to_hist_cur_pos_mean_n = scenario_stats.get_stats("dist_to_hist_cur_pos_mean")
    dist_to_hist_final_pos_n = scenario_stats.get_stats("dist_to_hist_final_pos")
    lane_change_n = scenario_stats.get_stats("lane_change")

    returns_n = {
        a_id: [sum(path[a_id]["rewards"]) for path in paths] for a_id in agent_ids
    }
    rewards_n = {
        a_id: np.concatenate([path[a_id]["rewards"] for path in paths])
        for a_id in agent_ids
    }
    actions_n = {a_id: [path[a_id]["actions"] for path in paths] for a_id in agent_ids}

    for a_id in agent_ids:
        for scenario_name, success_rate in success_rate_n[a_id].items():
            statistics[
                stat_prefix + f" {a_id} {scenario_name} Success Rate"
            ] = success_rate
        for scenario_name, collision_rate in collision_rate_n[a_id].items():
            statistics[
                stat_prefix + f" {a_id} {scenario_name} Collision Rate"
            ] = collision_rate
        for scenario_name, dist_to_hist_cur_pos_sum in dist_to_hist_cur_pos_sum_n[
            a_id
        ].items():
            statistics[
                stat_prefix + f" {a_id} {scenario_name} dist_to_hist_cur_pos_sum"
            ] = dist_to_hist_cur_pos_sum
        for scenario_name, dist_to_hist_cur_pos_mean in dist_to_hist_cur_pos_mean_n[
            a_id
        ].items():
            statistics[
                stat_prefix + f" {a_id} {scenario_name} dist_to_hist_cur_pos_mean"
            ] = dist_to_hist_cur_pos_mean
        for scenario_name, dist_to_hist_final_pos in dist_to_hist_final_pos_n[
            a_id
        ].items():
            statistics[
                stat_prefix + f" {a_id} {scenario_name} dist_to_hist_final_pos"
            ] = dist_to_hist_final_pos
        for scenario_name, lane_change in lane_change_n[a_id].items():
            statistics[
                stat_prefix + f" {a_id} {scenario_name} lane_change"
            ] = lane_change

        statistics.update(
            create_stats_ordered_dict(
                f"{a_id} Distance",
                distance_travelled_n[a_id],
                stat_prefix=stat_prefix,
                always_show_all_stats=True,
            )
        )
        statistics.update(
            create_stats_ordered_dict(
                f"{a_id} Rewards",
                rewards_n[a_id],
                stat_prefix=stat_prefix,
                always_show_all_stats=True,
            )
        )
        statistics.update(
            create_stats_ordered_dict(
                f"{a_id} Returns",
                returns_n[a_id],
                stat_prefix=stat_prefix,
                always_show_all_stats=True,
            )
        )
        statistics.update(
            create_stats_ordered_dict(
                f"{a_id} Actions",
                actions_n[a_id],
                stat_prefix=stat_prefix,
                always_show_all_stats=True,
            )
        )

    statistics.update(
        create_stats_ordered_dict(
            "Ep. Len.",
            np.array([len(path[agent_ids[0]]["terminals"]) for path in paths]),
            stat_prefix=stat_prefix,
            always_show_all_stats=True,
        )
    )
    statistics[stat_prefix + "Num Paths"] = len(paths)

    return statistics


def get_agent_mean_avg_returns(paths, std=False):
    agent_ids = paths[0].agent_ids
    n_agents = len(agent_ids)
    returns = [sum(path[a_id]["rewards"]) for path in paths for a_id in agent_ids]
    if std:
        return np.mean(returns) / n_agents, np.std(returns) / n_agents

    # take mean over multiple agents
    return np.mean(returns) / n_agents


def create_stats_ordered_dict(
    name,
    data,
    stat_prefix=None,
    always_show_all_stats=False,
    exclude_max_min=False,
):
    # print('\n<<<< STAT FOR {} {} >>>>'.format(stat_prefix, name))
    if stat_prefix is not None:
        name = "{} {}".format(stat_prefix, name)
    if isinstance(data, Number):
        # print('was a Number')
        return OrderedDict({name: data})

    if len(data) == 0:
        return OrderedDict()

    if isinstance(data, tuple):
        # print('was a tuple')
        ordered_dict = OrderedDict()
        for number, d in enumerate(data):
            sub_dict = create_stats_ordered_dict(
                "{0}_{1}".format(name, number),
                d,
            )
            ordered_dict.update(sub_dict)
        return ordered_dict

    if isinstance(data, list):
        # print('was a list')
        try:
            iter(data[0])
        except TypeError:
            pass
        else:
            data = np.concatenate(data)

    if isinstance(data, np.ndarray) and data.size == 1 and not always_show_all_stats:
        # print('was a numpy array of data.size==1')
        return OrderedDict({name: float(data)})

    # print('was a numpy array NOT of data.size==1')
    stats = OrderedDict(
        [
            (name + " Mean", np.mean(data)),
            (name + " Std", np.std(data)),
        ]
    )
    if not exclude_max_min:
        stats[name + " Max"] = np.max(data)
        stats[name + " Min"] = np.min(data)
    return stats


# I (Kamyar) will be adding my own eval utils here too
def plot_experiment_returns(
    exp_path,
    title,
    save_path,
    column_name="Test_Returns_Mean",
    x_axis_lims=None,
    y_axis_lims=None,
    constraints=None,
    plot_mean=False,
    plot_horizontal_lines_at=None,
    horizontal_lines_names=None,
):
    """
    plots the Test Returns Mean of all the
    """
    arr_list = []
    names = []

    dir_path = os.path.split(save_path)[0]
    os.makedirs(dir_path, exist_ok=True)

    # print(exp_path)

    for sub_exp_dir in os.listdir(exp_path):
        try:
            sub_exp_path = os.path.join(exp_path, sub_exp_dir)
            if not os.path.isdir(sub_exp_path):
                continue
            if constraints is not None:
                constraints_satisfied = True
                with open(os.path.join(sub_exp_path, "variant.json"), "r") as j:
                    d = json.load(j)
                for k, v in constraints.items():
                    k = k.split(".")
                    d_v = d[k[0]]
                    for sub_k in k[1:]:
                        d_v = d_v[sub_k]
                    if d_v != v:
                        constraints_satisfied = False
                        break
                if not constraints_satisfied:
                    # for debugging
                    # print('\nconstraints')
                    # print(constraints)
                    # print('\nthis dict')
                    # print(d)
                    continue

            csv_full_path = os.path.join(sub_exp_path, "progress.csv")
            # print(csv_full_path)
            try:
                progress_csv = np.genfromtxt(
                    csv_full_path, skip_header=0, delimiter=",", names=True
                )
                # print(progress_csv.dtype)
                if isinstance(column_name, str):
                    column_name = [column_name]
                for c_name in column_name:
                    if "+" in c_name:
                        first, second = c_name.split("+")
                        returns = progress_csv[first] + progress_csv[second]
                    elif "-" in c_name:
                        first, second = c_name.split("-")
                        returns = progress_csv[first] - progress_csv[second]
                    else:
                        returns = progress_csv[c_name]
                    arr_list.append(returns)
                    names.append(c_name + "_" + sub_exp_dir)
                # print(csv_full_path)
            except Exception:
                pass
        except Exception:
            pass

    if plot_mean:
        min_len = min(map(lambda a: a.shape[0], arr_list))
        arr_list = list(map(lambda a: a[:min_len], arr_list))
        returns = np.stack(arr_list)
        mean = np.mean(returns, 0)
        std = np.std(returns, 0)
        # save_plot(x, mean, title, save_path, color='cyan', x_axis_lims=x_axis_lims, y_axis_lims=y_axis_lims)
        plot_returns_on_same_plot(
            [mean, mean + std, mean - std],
            ["mean", "mean+std", "mean-std"],
            title,
            save_path,
            x_axis_lims=x_axis_lims,
            y_axis_lims=y_axis_lims,
        )
    else:
        if len(arr_list) == 0:
            print(0)
        if plot_horizontal_lines_at is not None:
            max_len = max(map(lambda a: a.shape[0], arr_list))
            arr_list += [np.ones(max_len) * y_val for y_val in plot_horizontal_lines_at]
            names += horizontal_lines_names
        try:
            # print(len(arr_list))
            plot_returns_on_same_plot(
                arr_list,
                names,
                title,
                save_path,
                x_axis_lims=x_axis_lims,
                y_axis_lims=y_axis_lims,
            )
        except Exception:
            print("Failed to plot:")
            print(arr_list)
            print(title)
            print(exp_path)
            print(constraints)
            # raise e
