from smarts_imitation.utils import common
import gym


def get_subscribed_features(neighbor_mode, closest_neighbor_num):
    if neighbor_mode == "LANE":
        subscribed_features = dict(
            ego_pos=(2,),
            heading=(1,),
            speed=(1,),
            neighbor_with_lanes=(closest_neighbor_num, 4),  # dist, speed, ttc
        )
    else:
        subscribed_features = dict(
            ego_dynamics=(1,),
            heading_errors=(3,),
        )
        if neighbor_mode == "RADIUS_EGO":
            subscribed_features["neighbor_with_radius_ego_coordinate"] = (
                closest_neighbor_num,
            )  # dist
        elif neighbor_mode == "RADIUS_WORLD":
            subscribed_features["neighbor_with_radius"] = (
                closest_neighbor_num * 4,
            )  # dist, speed, ttc
        else:
            raise NotImplementedError
    return subscribed_features


def get_observation_adapter(neighbor_mode, closest_neighbor_num=6):
    closest_neighbor_num = closest_neighbor_num
    img_resolution = 40
    observe_lane_num = 3

    subscribed_features = get_subscribed_features(
        neighbor_mode, closest_neighbor_num)

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
