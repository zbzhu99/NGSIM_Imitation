from rlkit.env_creators.particle.particle_env import ParticleEnv
from rlkit.env_creators.mujoco.mujoco_env import MujocoEnv
from rlkit.env_creators.mpe.mpe_env import MpeEnv
from rlkit.env_creators.gym.gym_env import GymEnv
from rlkit.env_creators.smarts.smarts_env import SmartsEnv
from rlkit.env_creators.masmarts.masmarts_env import MASmartsEnv
from rlkit.env_creators.ppuu.ppuu_env import PPUUEnv


def get_env_cls(env_creator_name: str):
    return {
        "mujoco": MujocoEnv,
        "particle": ParticleEnv,
        "mpe": MpeEnv,
        "gym": GymEnv,
        "smarts": SmartsEnv,
        "masmarts": MASmartsEnv,
        "ppuu": PPUUEnv,
    }[env_creator_name]
