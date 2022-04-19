import yaml
import argparse
import joblib
import os
import sys
import gym
import inspect
import pickle
import numpy as np
import signal
import random
from subprocess import Popen

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
print(sys.path)

from rlkit.envs import get_env
from smarts.core.smarts import SMARTS
import rlkit.torch.utils.pytorch_util as ptu
from rlkit.launchers.launcher_util import set_seed
from rlkit.envs.wrappers import ProxyEnv, NormalizedBoxActEnv, ObsScaledEnv
from rlkit.torch.common.policies import MakeDeterministic
from smarts.core.scenario import Scenario
from envision.client import Client as Envision
from smarts_imitation import ScenarioZoo


def experiment():
    scenario_name = "ngsim_i80"
    scenarios = ScenarioZoo.get_scenario(scenario_name)
    scenario_iterator = Scenario.scenario_variations(
        [scenarios], list([]), shuffle_scenarios=False, circular=False
    )  # scenarios with different traffic histories.
    scenario = next(scenario_iterator)

    envision_client = Envision(
        sim_name="ngsim_us101",
        output_dir="record_data_replay_path",
        headless=False,
    )
    smarts = SMARTS(
        agent_interfaces={},
        traffic_sim=None,
        envision=envision_client,
    )
    smarts.reset(scenario)
    smarts.step({})
    while True:
        smarts.step({})
        current_vehicles = smarts.vehicle_index.social_vehicle_ids()
        if len(current_vehicles) == 0:
            break


if __name__ == "__main__":
    # Arguments

    envision_proc = Popen(
        f"scl envision start -s {ScenarioZoo.get_scenario('ngsim_i80')} -p 8081",
        shell=True,
        preexec_fn=os.setsid,
    )

    try:
        experiment()
    except Exception as e:
        os.killpg(os.getpgid(envision_proc.pid), signal.SIGTERM)
        envision_proc.wait()
        raise e

    os.killpg(os.getpgid(envision_proc.pid), signal.SIGTERM)
    envision_proc.wait()
