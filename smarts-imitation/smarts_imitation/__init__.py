import os
import numpy as np
from pathlib import Path

from gym.envs.registration import register


NGSIM_SCENARIO_PATH = str(Path(__file__).resolve().parent.parent / "ngsim")

register(
    id="SMARTS-Imitation-v0",
    entry_point="smarts_imitation.envs:SMARTSImitation",
    kwargs=dict(
        scenarios=[
            os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "ngsim",
            )
        ],
        action_range=np.array(
            [
                [-8.0, -2.5],
                [8.0, 2.5],
            ]
        ),
    ),
)
