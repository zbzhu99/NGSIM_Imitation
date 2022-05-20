import json
import numpy as np
from smarts_imitation.utils.common import CalObs, _legalize_angle
import math
from smarts.core.utils.math import vec_to_radians, radians_to_vec


class SingleVehicleState:
    def __init__(
        self,
        heading,
        position,
        speed,
    ):
        # heading
        self.heading = heading

        # position
        position += [0.0]
        self.position = np.array(position)

        # speed
        assert isinstance(speed, (int, float))
        self.speed = speed

    def __str__(self):
        return (
            f"heading: {self.heading}, position: {self.position}, speed: {self.speed}"
        )


class WayPoint:
    def __init__(
        self,
        pos,
        lane_index,
        heading,
    ):
        # pos
        pos += [0]
        self.pos = np.array(pos)

        # lane_index
        assert isinstance(lane_index, int)
        self.lane_index = lane_index

        # heading
        assert isinstance(lane_index, (int, float))
        self.heading = heading

    def __str__(self):
        return (
            f"pos: {self.pos}, lane_index: {self.lane_index}, heading: {self.heading}"
        )

    def dist_to(self, position):
        assert isinstance(position, np.ndarray)
        return np.linalg.norm(position - self.pos)


class EnvObs:
    def __init__(
        self,
        ego_vehicle_state,
        neighborhood_vehicle_states,
        waypoint_paths,
    ):
        self.ego_vehicle_state = ego_vehicle_state
        self.neighborhood_vehicle_states = neighborhood_vehicle_states
        self.waypoint_paths = waypoint_paths


def build_raw_obs(raw_data):
    ego_vehicle_state = build_ego_vehicle(raw_data)
    print(f"ego_vehicle_state: {ego_vehicle_state}")
    neighborhood_vehicle_states = build_neighbor_vehicle(raw_data)

    print(f"len neighbor: {len(neighborhood_vehicle_states)}")
    for neighbor in neighborhood_vehicle_states:
        print(f"neighbor: {neighbor}")
    waypoint_paths = build_waypoint_paths(raw_data, ego_vehicle_state.position)

    env_obs = EnvObs(
        ego_vehicle_state=ego_vehicle_state,
        neighborhood_vehicle_states=neighborhood_vehicle_states,
        waypoint_paths=waypoint_paths,
    )
    return env_obs


def build_single_vehicle(vehicle_info):
    assert len(vehicle_info) == 4, f"vehicle_info: {vehicle_info}"
    vehicle_pos = vehicle_info[:2]

    vehicle_heading = vehicle_info[2]
    vehicle_heading -= math.pi / 2
    vehicle_heading = _legalize_angle(vehicle_heading)

    vehicle_speed = vehicle_info[3]

    vehicle_state = SingleVehicleState(
        heading=vehicle_heading,
        position=vehicle_pos,
        speed=vehicle_speed,
    )
    return vehicle_state


def build_single_waypoint_path(lane_index, route):
    waypoint_path = []
    num = len(route)
    print(f"lane_index {lane_index}, number waypoints: {num}")
    for i in range(1, num):
        prev_pos = route[i - 1]
        cur_pos = route[i]
        cur_heading_vec = [cur_pos[0] - prev_pos[0], cur_pos[1] - prev_pos[1]]
        cur_heading = vec_to_radians(cur_heading_vec)
        waypoint = WayPoint(
            pos=cur_pos,
            lane_index=lane_index,
            heading=cur_heading,
        )
        waypoint_path.append(waypoint)
    return waypoint_path


def build_waypoint_paths(raw_data, ego_position):
    waypoint_paths = []

    routes = raw_data["routes"]
    route_names = {i: f"route{i}" for i in [1, 2, 3]}
    for route_index, route_name in route_names.items():
        route_i = routes[route_name]
        waypoint_path_i = build_single_waypoint_path(route_index, route_i)
        waypoint_path_i = sorted(
            waypoint_path_i, key=lambda wp: wp.dist_to(ego_position)
        )
        waypoint_paths.append(waypoint_path_i)
    return waypoint_paths


def build_ego_vehicle(raw_data):
    ego_info = raw_data["ego_info"]
    ego_vehicle_state = build_single_vehicle(ego_info)
    return ego_vehicle_state


def build_neighbor_vehicle(raw_data):
    neighbor_vehicle_states = []
    obj_infos = raw_data["obj_infos"]
    for obj_info in obj_infos:
        neighbor_state = build_single_vehicle(obj_info)
        neighbor_vehicle_states.append(neighbor_state)
    return neighbor_vehicle_states


def load_json(path):
    with open(path, "rb") as f:
        data = json.load(f)
    return data


def test(path):
    raw_data = load_json(path)
    print(raw_data)
    env_obs = build_raw_obs(raw_data)

    heading = CalObs.cal_ego_dynamics(env_obs)
    print(f"heading: {heading}")
    lane_errors = CalObs.cal_lane_errors(env_obs)
    print(f"lane_errors: {lane_errors}")
    neighbor = CalObs.cal_neighbor_with_radius_ego_coordinate(env_obs)
    print(f"neighbor: {neighbor}")

    data = np.concatenate([heading, lane_errors, neighbor]).tolist()
    js = {"feature": data}
    js = json.dumps(js)
    with open("feature.json", "w") as f:
        f.write(js)


if __name__ == "__main__":

    path = "./test_raw_data.json"
    test(path)
