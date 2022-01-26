import os
import sys
import pickle
import inspect
import argparse
import matplotlib

matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "demo_path",
        type=str,
        help="Path to the demo file.",
    )
    args = parser.parse_args()

    with open(args.demo_path, "rb") as f:
        path_builders = pickle.load(f)

    total_observations = []
    total_actions = []

    for path_builder in path_builders:
        observations = np.stack(path_builder["agent_0"]["observations"], axis=0)
        actions = np.stack(path_builder["agent_0"]["actions"], axis=0)
        total_observations.append(observations)
        total_actions.append(actions)

    total_observations = np.concatenate(total_observations, axis=0)
    total_actions = np.concatenate(total_actions, axis=0)

    print("Action Max: {}".format(total_actions.max(axis=0)))
    print("Action Min: {}".format(total_actions.min(axis=0)))
    print("Action Mean: {}".format(total_actions.mean(axis=0)))
    print("Action Std: {}".format(total_actions.std(axis=0)))

    plt.figure()
    plt.hist(total_actions[:, 0], bins=500)
    plt.xlim(-3, 3)
    plt.savefig("action_dim_0.png")
    plt.figure()
    plt.hist(total_actions[:, 1], bins=500)
    plt.xlim(-1.5, 1.5)
    plt.savefig("action_dim_1.png")
