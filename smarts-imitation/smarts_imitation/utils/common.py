"""
This file implements the calculation of available features independently. For usage, you should call
`subscribe_features` firstly, then retrive the corresponding observation adapter by define observation space

observation_space = gym.spaces.Dict(subscribe_features(`
    dict(
        distance_to_center=(stack_size, 1),
        speed=(stack_size, 1),
        steering=(stack_size, 1),
        heading_errors=(stack_size, look_ahead),
        ego_lane_dist_and_speed=(stack_size, observe_lane_num + 1),
        img_gray=(stack_size, img_resolution, img_resolution),
    )
))

obs_adapter = get_observation_adapter(
    observation_space,
    look_ahead=look_ahead,
    observe_lane_num=observe_lane_num,
    resize=(img_resolution, img_resolution),
)

"""

import math
import gym
import cv2
import numpy as np

from smarts.core.sensors import Observation
from smarts.core.utils.math import vec_2d, vec_to_radians, radians_to_vec


def _legalize_angle(angle):
    """Return the angle within range [0, 2pi)"""
    return angle % (2 * math.pi)


def _get_closest_vehicles(ego, neighbor_vehicles, n):
    """将周角分成n个区域, 获取每个区域最近的车辆"""
    ego_pos = ego.position[:2]
    groups = {i: (None, 1e10) for i in range(n)}
    partition_size = math.pi * 2.0 / n
    # get partition
    for v in neighbor_vehicles:
        v_pos = v.position[:2]
        rel_pos_vec = np.array([v_pos[0] - ego_pos[0], v_pos[1] - ego_pos[1]])
        # calculate its partitions
        angle = vec_to_radians(rel_pos_vec)
        i = int(angle / partition_size)
        dist = np.sqrt(rel_pos_vec.dot(rel_pos_vec))
        if dist < groups[i][1]:
            groups[i] = (v, dist)

    return groups


def _get_closest_vehicles_with_ego_coordinate(ego, neighbor_vehicles, n):
    """将周角分成n个区域, 获取每个区域最近的车辆"""
    ego_pos = ego.position[:2]

    groups = {i: (None, float("inf")) for i in range(n)}
    partition_size = math.pi * 2.0 / n
    # get partition
    for v in neighbor_vehicles:
        v_pos = v.position[:2]
        rel_pos_vec = np.array([v_pos[0] - ego_pos[0], v_pos[1] - ego_pos[1]])
        # calculate its partitions
        angle = vec_to_radians(rel_pos_vec)
        rel_angle = angle - ego.heading
        rel_angle = _legalize_angle(rel_angle)
        rel_angle = _legalize_angle(rel_angle + partition_size / 2.0)
        i = int(rel_angle / partition_size)
        dist = np.sqrt(rel_pos_vec.dot(rel_pos_vec))
        if dist < groups[i][1]:
            groups[i] = (v, dist)

    return groups


def _get_relative_position(ego, neighbor_vehicle, relative_lane):
    relative_pos = 1 if ego.position[0] > neighbor_vehicle.position[0] else 0
    return relative_pos * 3 + relative_lane + 1


class CalObs:
    @staticmethod
    def get_feature_space(feature_name, **kwargs):
        if not hasattr(CalObs, "_spaces"):
            CalObs._spaces = dict(
                ego_dynamics=gym.spaces.Box(low=-330, high=330, shape=(1,)),
                ego_pos=gym.spaces.Box(low=-1e3, high=1e3, shape=(2,)),
                heading=gym.spaces.Box(low=-1e3, high=1e3, shape=(1,)),
                distance_to_center=gym.spaces.Box(low=-1e3, high=1e3, shape=(1,)),
                lane_errors=gym.spaces.Box(low=-1.0, high=1.0, shape=(3 * 5,)),
                speed=gym.spaces.Box(low=-330.0, high=330.0, shape=(1,)),
                steering=gym.spaces.Box(low=-1.0, high=1.0, shape=(1,)),
                neighbor_with_radius=gym.spaces.Box(
                    low=-1e3, high=1e3, shape=(kwargs.get("closest_neighbor_num") * 4,)
                ),
                neighbor_with_radius_ego_coordinate=gym.spaces.Box(
                    low=-1e3, high=1e3, shape=(kwargs.get("closest_neighbor_num") * 4,)
                ),
                neighbor_with_lanes=gym.spaces.Box(
                    low=-1e3, high=1e3, shape=(kwargs.get("closest_neighbor_num") * 4,)
                ),
            )
        return CalObs._spaces[feature_name]

    @staticmethod
    def cal_ego_dynamics(env_obs: Observation, **kwargs):
        ego = env_obs.ego_vehicle_state
        ego_dynamics = np.array([float(ego.speed)], dtype="float64").reshape((-1,))
        return ego_dynamics

    @staticmethod
    def cal_ego_pos(env_obs: Observation, **kwargs):
        return env_obs.ego_vehicle_state.position[:2]

    @staticmethod
    def cal_heading(env_obs: Observation, **kwargs):
        return np.asarray(float(env_obs.ego_vehicle_state.heading))

    @staticmethod
    def cal_distance_to_center(env_obs: Observation, **kwargs):
        """Calculate the signed distance to the center of the current lane.
        Return a FeatureMetaInfo(space, data) instance
        """

        ego = env_obs.ego_vehicle_state
        waypoint_paths = env_obs.waypoint_paths
        wps = [path[0] for path in waypoint_paths]
        closest_wp = min(wps, key=lambda wp: wp.dist_to(ego.position))
        signed_dist_to_center = closest_wp.signed_lateral_error(ego.position)
        lane_width = closest_wp.lane_width * 0.5
        # TODO(ming): for the case of overwhilm, it will throw error
        norm_dist_from_center = signed_dist_to_center / lane_width

        dist = np.asarray([norm_dist_from_center])
        return dist

    @staticmethod
    def cal_lane_errors(env_obs: Observation, **kwargs):
        # look_ahead = kwargs["look_ahead"]
        look_ahead = 3
        ego = env_obs.ego_vehicle_state
        waypoint_paths = env_obs.waypoint_paths
        wps = [path[0] for path in waypoint_paths]
        closest_wp = min(wps, key=lambda wp: wp.dist_to(ego.position))
        closest_wps = waypoint_paths[closest_wp.lane_index][:look_ahead]

        features = []
        for closest_wp in closest_wps:
            lane_rel_heading_radians = _legalize_angle(closest_wp.heading - ego.heading)
            lane_rel_heading_vec = radians_to_vec(lane_rel_heading_radians)
            wp_rel_pos = [
                closest_wp.pos[0] - ego.position[0],
                closest_wp.pos[1] - ego.position[1],
            ]
            wp_rel_heading_radians = vec_to_radians(wp_rel_pos)
            wp_rel_heading_vec = radians_to_vec(
                _legalize_angle(wp_rel_heading_radians - ego.heading)
            )
            wp_dist = np.linalg.norm(wp_rel_pos)

            wp_error = [
                lane_rel_heading_vec[0],
                lane_rel_heading_vec[1],
                wp_rel_heading_vec[0],
                wp_rel_heading_vec[1],
                wp_dist,
            ]
            features.append(wp_error)
        for _ in range(look_ahead - len(features)):
            wp_error = [0, 0, 0, 0, -1]
            features.append(wp_error)

        return np.concatenate(features, axis=-1)

    @staticmethod
    def cal_speed(env_obs: Observation, **kwargs):
        ego = env_obs.ego_vehicle_state
        res = np.asarray([ego.speed])
        return res

    @staticmethod
    def cal_steering(env_obs: Observation, **kwargs):
        ego = env_obs.ego_vehicle_state
        return np.asarray([ego.steering / 45.0])

    @staticmethod
    def cal_neighbor_with_radius(env_obs: Observation, **kwargs):
        ego = env_obs.ego_vehicle_state
        neighbor_vehicle_states = env_obs.neighborhood_vehicle_states
        closest_neighbor_num = kwargs.get("closest_neighbor_num", 8)
        # dist, speed, ttc, pos
        features = np.zeros((closest_neighbor_num, 4))
        # fill neighbor vehicles into closest_neighboor_num areas
        surrounding_vehicles = _get_closest_vehicles(
            ego, neighbor_vehicle_states, n=closest_neighbor_num
        )
        for i, v in surrounding_vehicles.items():
            if v[0] is None:
                v = ego
            else:
                v = v[0]

            pos = v.position[:2]
            heading = np.asarray(float(v.heading))
            speed = np.asarray(v.speed)

            features[i, :] = np.asarray([pos[0], pos[1], heading, speed])

        return features.reshape((-1,))

    @staticmethod
    def cal_neighbor_with_radius_ego_coordinate(env_obs: Observation, **kwargs):
        ego = env_obs.ego_vehicle_state
        neighbor_vehicle_states = env_obs.neighborhood_vehicle_states
        closest_neighbor_num = kwargs.get("closest_neighbor_num", 8)
        # dist, speed, ttc, pos
        features = [None] * closest_neighbor_num
        # get the closest vehicles according to the ego coordinate system
        surrounding_vehicles = _get_closest_vehicles_with_ego_coordinate(
            ego, neighbor_vehicle_states, n=closest_neighbor_num
        )
        for i, v in surrounding_vehicles.items():
            if v[0] is None:
                # dist = 15.0
                dist = -5.0
                heading_vec = [0.0, 0.0]
                speed = -1.0
            else:
                # dist = min(v[1], 15.0)
                dist = v[1]
                heading_rad = _legalize_angle(v[0].heading - ego.heading)
                heading_vec = radians_to_vec(heading_rad)
                speed = v[0].speed
            features[i] = np.array(
                [dist, heading_vec[0], heading_vec[1], speed], dtype="float64"
            )
        features = np.concatenate(features, axis=-1)

        return features

    @staticmethod
    def cal_neighbor_with_lanes(env_obs: Observation, **kwargs):
        """计算自车道以及左右相邻车道车辆，每个车道前后各一辆"""
        ego = env_obs.ego_vehicle_state
        waypoint_paths = env_obs.waypoint_paths
        wps = [path[0] for path in waypoint_paths]
        closest_wp = min(wps, key=lambda wp: wp.dist_to(ego.position))
        closest_neighbor_num = kwargs.get("closest_neighbor_num", 8)
        # dist, speed, ttc, pos
        features = np.zeros((closest_neighbor_num, 4))

        wps_with_lane_dist = []
        for path_idx, path in enumerate(waypoint_paths):
            wps_with_lane_dist.append((path[-1], path_idx))

        wps_on_lane = [
            (wp, path_idx)
            for wp, path_idx in wps_with_lane_dist
            # if wp.lane_id == v.lane_id
        ]

        ego_lane_index = closest_wp.lane_index

        surrounding_vehicles = {i: (None, 1e10) for i in range(6)}

        for v in env_obs.neighborhood_vehicle_states:
            nearest_wp, path_idx = min(
                wps_on_lane,
                key=lambda tup: np.linalg.norm(tup[0].pos - vec_2d(v.position)),
            )
            if abs(ego_lane_index - path_idx) > 1:
                # not in neighbor lane
                continue
            rel_pos_vec = np.asarray(
                [v.position[0] - ego.position[0], v.position[1] - ego.position[1]]
            )
            pos_id = _get_relative_position(ego, v, ego_lane_index - path_idx)
            dist = np.sqrt(rel_pos_vec.dot(rel_pos_vec))
            if dist < surrounding_vehicles[pos_id][1]:
                surrounding_vehicles[pos_id] = (v, dist)

        for i, v in surrounding_vehicles.items():
            if v[0] is None:
                v = ego
            else:
                v = v[0]

            pos = v.position[:2]
            heading = np.asarray(float(v.heading))
            speed = np.asarray(v.speed)

            features[i, :] = np.asarray([pos[0], pos[1], heading, speed])

        return features.reshape((-1,))

    @staticmethod
    def cal_ego_lane_dist_and_speed(env_obs: Observation, **kwargs):
        """Calculate the distance from ego vehicle to its front vehicles (if have) at observed lanes,
        also the relative speed of the front vehicle which positioned at the same lane.
        """
        observe_lane_num = kwargs["observe_lane_num"]
        ego = env_obs.ego_vehicle_state
        waypoint_paths = env_obs.waypoint_paths
        wps = [path[0] for path in waypoint_paths]
        closest_wp = min(wps, key=lambda wp: wp.dist_to(ego.position))

        wps_with_lane_dist = []
        for path_idx, path in enumerate(waypoint_paths):
            lane_dist = 0.0
            for w1, w2 in zip(path, path[1:]):
                wps_with_lane_dist.append((w1, path_idx, lane_dist))
                lane_dist += np.linalg.norm(w2.pos - w1.pos)
            wps_with_lane_dist.append((path[-1], path_idx, lane_dist))

        # TTC calculation along each path
        ego_closest_wp = min(wps, key=lambda wp: wp.dist_to(ego.position))

        wps_on_lane = [
            (wp, path_idx, dist)
            for wp, path_idx, dist in wps_with_lane_dist
            # if wp.lane_id == v.lane_id
        ]

        ego_lane_index = closest_wp.lane_index
        lane_dist_by_path = [1] * len(waypoint_paths)
        ego_lane_dist = [0] * observe_lane_num
        speed_of_closest = 0.0

        for v in env_obs.neighborhood_vehicle_states:
            nearest_wp, path_idx, lane_dist = min(
                wps_on_lane,
                key=lambda tup: np.linalg.norm(tup[0].pos - vec_2d(v.position)),
            )
            if np.linalg.norm(nearest_wp.pos - vec_2d(v.position)) > 2:
                # this vehicle is not close enough to the path, this can happen
                # if the vehicle is behind the ego, or ahead past the end of
                # the waypoints
                continue

            # relative_speed_m_per_s = (ego.speed - v.speed) * 1000 / 3600
            # relative_speed_m_per_s = max(abs(relative_speed_m_per_s), 1e-5)
            dist_wp_vehicle_vector = vec_2d(v.position) - vec_2d(nearest_wp.pos)
            direction_vector = radians_to_vec.dot(dist_wp_vehicle_vector)

            dist_to_vehicle = lane_dist + np.sign(direction_vector) * (
                np.linalg.norm(vec_2d(nearest_wp.pos) - vec_2d(v.position))
            )
            lane_dist = dist_to_vehicle / 100.0

            if lane_dist_by_path[path_idx] > lane_dist:
                if ego_closest_wp.lane_index == v.lane_index:
                    speed_of_closest = (v.speed - ego.speed) / 120.0

            lane_dist_by_path[path_idx] = min(lane_dist_by_path[path_idx], lane_dist)

        # current lane is centre
        flag = observe_lane_num // 2
        ego_lane_dist[flag] = lane_dist_by_path[ego_lane_index]

        max_lane_index = len(lane_dist_by_path) - 1

        if max_lane_index == 0:
            right_sign, left_sign = 0, 0
        else:
            right_sign = -1 if ego_lane_index + 1 > max_lane_index else 1
            left_sign = -1 if ego_lane_index - 1 >= 0 else 1

        ego_lane_dist[flag + right_sign] = lane_dist_by_path[
            ego_lane_index + right_sign
        ]
        ego_lane_dist[flag + left_sign] = lane_dist_by_path[ego_lane_index + left_sign]

        res = np.asarray(ego_lane_dist + [speed_of_closest])
        return res
        # space = SPACE_LIB["goal_relative_pos"](res.shape)
        # return (res - space.low) / (space.high - space.low)

    @staticmethod
    def cal_img_gray(env_obs: Observation, **kwargs):
        resize = kwargs["resize"]

        def rgb2gray(rgb):
            return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

        rgb_ndarray = env_obs.top_down_rgb
        gray_scale = (
            cv2.resize(
                rgb2gray(rgb_ndarray), dsize=resize, interpolation=cv2.INTER_CUBIC
            )
            / 255.0
        )
        return gray_scale

    @staticmethod
    def cal_neighbor_dict(env_obs: Observation, **kwargs):
        ego = env_obs.ego_vehicle_state
        neighbor_vehicle_states = env_obs.neighborhood_vehicle_states
        closest_neighbor_num = kwargs.get("closest_neighbor_num", 8)
        surrounding_vehicles = _get_closest_vehicles(
            ego, neighbor_vehicle_states, n=closest_neighbor_num
        )
        neighbor_dict = gym.spaces.Dict()
        for i, v in surrounding_vehicles.items():
            if v[0] is None:
                v = ego
            else:
                v = v[0]

            pos = v.position[:2]
            heading = np.asarray(float(v.heading))
            speed = np.asarray(v.speed)

            neighbor_dict[v.id] = np.asarray([pos[0], pos[1], heading, speed])

        return neighbor_dict


def _update_obs_by_item(
    obs_placeholder: dict, tuned_obs: dict, space_dict: gym.spaces.Dict
):
    for key, value in tuned_obs.items():
        if isinstance(value, gym.spaces.Dict):
            obs_placeholder[key] = gym.spaces.Dict()
            for k, v in value.spaces.items():
                obs_placeholder[key][k] = v
            continue
        if obs_placeholder.get(key, None) is None:
            obs_placeholder[key] = np.zeros(space_dict[key].shape)
        obs_placeholder[key] = value.reshape(space_dict[key].shape)


def _cal_obs(env_obs: Observation, space, **kwargs):
    obs = dict()
    for name in space.spaces:
        if hasattr(CalObs, f"cal_{name}"):
            obs[name] = getattr(CalObs, f"cal_{name}")(env_obs, **kwargs)
    return obs


def subscribe_features(feature_list, **kwargs):
    res = dict()
    for feature_name in feature_list:
        res[feature_name] = CalObs.get_feature_space(feature_name, **kwargs)
    return res


# XXX(ming): refine it as static method
def get_observation_adapter(observation_space, **kwargs):
    def observation_adapter(env_obs):
        obs = dict()
        temp = _cal_obs(env_obs, observation_space, **kwargs)
        _update_obs_by_item(obs, temp, observation_space)
        return obs

    return observation_adapter


def default_info_adapter(shaped_reward: float, raw_info: dict):
    return raw_info
