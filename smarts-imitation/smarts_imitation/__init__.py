import numpy as np
from pathlib import Path

from smarts_imitation.envs import SMARTSImitation


class ScenarioZoo:
    _scenarios = {
        "NGSIM-I80": str(Path(__file__).resolve().parent.parent / "ngsim_i80"),
    }

    @classmethod
    def get_scenario(cls, scenario_name):
        if scenario_name not in cls._scenarios:
            raise ValueError(
                scenario_name, f"{scenario_name} is not in the scenario zoo."
            )
        return cls._scenarios[scenario_name]


def create_env(scenario_name, **kwargs):
    if scenario_name == "NGSIM-I80":
        return SMARTSImitation(
            scenarios=[ScenarioZoo.get_scenario("NGSIM-I80")],
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
