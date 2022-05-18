import argparse
import os
import sys
import inspect
import pickle
import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
print(sys.path)

COLOR_LIST = ["red", "blue", "green", "black", "pink"]
X_LIM = [0, 430]
Y_LIM = [3, 25]
FMT = "-"
LINE_WIDTH = 1
FIGURE_SIZE = (150, 10)
ALPHA = 0.5

x_min = np.inf
y_min = np.inf
x_max = -np.inf
y_max = -np.inf


def load_trajectories(trajectory_save_dir):
    file_names = []
    for name in os.listdir(trajectory_save_dir):
        if not name.endswith(".pkl"):
            continue
        file_names.append(name)
    trajectories = {}
    for file_name in file_names:
        name = "".join(file_name.split(".")[:-1])
        path = os.path.join(trajectory_save_dir, file_name)
        with open(path, "rb") as f:
            traj = pickle.load(f)
            trajectories[name] = traj
    return trajectories


def draw_single_case(traj, traj_name, images_save_dir):
    plt.rcParams["figure.figsize"] = FIGURE_SIZE
    plt.clf()
    image_name = "-".join([str(x) for x in traj_name])
    image_path = os.path.join(images_save_dir, image_name)
    legend = []
    for i, model_name in enumerate(traj.keys()):
        t = traj[model_name]
        if len(t) == 0:
            continue
        color_i = COLOR_LIST[i]
        legend.append(model_name)
        t = np.array(t)
        assert len(t.shape) == 2
        x_i = t[:, 0]
        y_i = t[:, 1]

        # compute min max
        global x_min, x_max, y_min, y_max
        x_min = min(np.min(x_i), x_min)
        y_min = min(np.min(y_i), y_min)
        x_max = max(np.max(x_i), x_max)
        y_max = max(np.max(y_i), y_max)

        plt.plot(x_i, y_i, FMT, color=color_i, linewidth=LINE_WIDTH, alpha=ALPHA)
    plt.xlim(X_LIM)
    plt.ylim(Y_LIM)
    plt.legend(legend)
    plt.savefig(image_path)


def draw_trajectories(trajectories, images_save_dir):
    if not os.path.exists(images_save_dir):
        os.makedirs(images_save_dir)
    model_names = sorted(list(trajectories.keys()))

    traj_names = list(trajectories[model_names[0]].keys())
    for traj_name in traj_names:
        traj = OrderedDict()
        for model_name in model_names:
            if traj_name in trajectories[model_name]:
                traj[model_name] = trajectories[model_name][traj_name]
            else:
                traj[model_name] = []
        draw_single_case(traj, traj_name, images_save_dir)
    print("finished!")


def experiment(trajectory_save_dir):
    trajectories = load_trajectories(trajectory_save_dir)
    draw_trajectories(
        trajectories, images_save_dir=os.path.join(trajectory_save_dir, "images/")
    )
    print(f"x_min: {x_min}, x_max: {x_max}, y_min: {y_min}, y_max: {y_max}")


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--trajectory_save_dir", help="trajectory_save_dir")
    args = parser.parse_args()
    experiment(args.trajectory_save_dir)
