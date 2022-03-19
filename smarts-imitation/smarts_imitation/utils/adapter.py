from smarts_imitation.utils import common
import gym


def get_observation_adapter(neighbor_mode="LANE"):
    closest_neighbor_num = 6
    img_resolution = 40
    observe_lane_num = 3
    subscribed_features = dict(
        ego_pos=(2,),
        heading=(1,),
        speed=(1,),
    )

    if neighbor_mode == "RADIUS":  # Neighbor vehicles are selected by radius around the ego.
        subscribed_features["neighbor_with_radius"] = (closest_neighbor_num * 4,)  # dist, speed, ttc
    elif neighbor_mode == "LANE":  # Neighbor vehicles are selected from adjacent lanes.
        subscribed_features["neighbor_with_lanes"] = (closest_neighbor_num * 4,)  # dist, speed, ttc
    else:
        raise NotImplementedError

    observation_space = gym.spaces.Dict(
        common.subscribe_features(**subscribed_features)
    )

    observation_adapter = common.get_observation_adapter(
        observation_space,
        observe_lane_num=observe_lane_num,
        resize=(img_resolution, img_resolution),
        closest_neighbor_num=closest_neighbor_num,
    )

    return observation_adapter


def get_action_adapter():
    def action_adapter(model_action):
        assert len(model_action) == 2
        return (model_action[0], model_action[1])

    return action_adapter
