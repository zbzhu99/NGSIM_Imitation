from smarts.zoo.agent_spec import AgentSpec
from smarts.core.agent_interface import AgentInterface
from smarts.core.controllers import ActionSpaceType
from smarts.core.agent_interface import DoneCriteria
from smarts_imitation.utils import adapter


def get_agent_spec(feature_list, closest_neighbor_num, collision_done=True):
    done_criteria = DoneCriteria(
        collision=collision_done,
        off_road=collision_done,
        off_route=collision_done,
    )
    interface = AgentInterface(
        done_criteria=done_criteria,
        max_episode_steps=None,
        waypoints=True,
        neighborhood_vehicles=True,
        ogm=False,
        rgb=False,
        lidar=False,
        action=ActionSpaceType.Imitation,
    )
    agent_spec = AgentSpec(
        interface=interface,
        observation_adapter=adapter.get_observation_adapter(
            feature_list=feature_list,
            closest_neighbor_num=closest_neighbor_num,
        ),
        action_adapter=adapter.get_action_adapter(),
    )

    return agent_spec
