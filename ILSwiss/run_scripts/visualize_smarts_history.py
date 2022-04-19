import os
import sys
import inspect
import signal
from subprocess import Popen

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
print(sys.path)

from smarts.core.smarts import SMARTS
from smarts.core.scenario import Scenario
from envision.client import Client as Envision
from smarts_imitation import ScenarioZoo


def experiment(scenario_path, scenario_name, traffic_name):

    scenario_iterator = Scenario.scenario_variations(
        [scenario_path], list([]), shuffle_scenarios=False, circular=False
    )  # scenarios with different traffic histories.
    scenario = None
    for _scenario in scenario_iterator:
        if _scenario._traffic_history.name == traffic_name:
            scenario = _scenario
            break
    assert scenario is not None

    envision_client = Envision(
        sim_name=f"{scenario_name}_{traffic_name}",
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

    scenario_name = "ngsim_us101"
    traffic_name = "us101_0750-0805"
    scenario_path = ScenarioZoo.get_scenario(scenario_name)

    # Arguments
    envision_proc = Popen(
        f"scl envision start -s {scenario_path} -p 8081",
        shell=True,
        preexec_fn=os.setsid,
    )

    try:
        experiment(scenario_path, scenario_name, traffic_name)
    except Exception as e:
        os.killpg(os.getpgid(envision_proc.pid), signal.SIGTERM)
        envision_proc.wait()
        raise e

    os.killpg(os.getpgid(envision_proc.pid), signal.SIGTERM)
    envision_proc.wait()
