import numpy as np
from pathlib import Path

from smarts_imitation.envs import SMARTSImitation


class ScenarioZoo:
    _scenarios = {
        "ngsim_i80": str(
            Path(__file__).resolve().parent.parent / "scenarios" / "ngsim_i80"
        ),
        "ngsim_us101": str(
            Path(__file__).resolve().parent.parent / "scenarios" / "ngsim_us101"
        ),
    }

    @classmethod
    def get_scenario(cls, scenario_name):
        if scenario_name not in cls._scenarios:
            raise ValueError(
                scenario_name, f"{scenario_name} is not in the scenario zoo."
            )
        return cls._scenarios[scenario_name]


def create_env(scenario_name, traffic_name, **kwargs):
    if scenario_name == "ngsim_i80":
        return SMARTSImitation(
            scenario_path=ScenarioZoo.get_scenario("ngsim_i80"),
            traffic_name=traffic_name,
            action_range=np.array(
                [
                    [-8.0, -2.5],
                    [8.0, 2.5],
                ]
            ),
            **kwargs,
        )
    elif scenario_name == "ngsim_us101":
        return SMARTSImitation(
            scenario_path=ScenarioZoo.get_scenario("ngsim_us101"),
            traffic_name=traffic_name,
            action_range=np.array(
                [
                    [-8.0, -2.5],
                    [8.0, 2.5],
                ]
            ),
            **kwargs,
        )
    else:
        raise ValueError(f"{scenario_name} is not in the scenario zoo.")
