import os
import torch
import pickle
import numpy as np


train_split_path = "/NAS2020/Workspaces/DRLGroup/zbzhu/lfo-ppuu/lfo/demos/expert_trajs_50/PPUU/train_indx_final.pkl"
with open(train_split_path, "rb") as f:
    train_index_dict = pickle.load(f)
train_car_idx = list(train_index_dict.keys())

data_dir = "traffic-data/state-action-cost/data_i80_v0"
data_files = next(os.walk(data_dir))[1]
stats_path = data_dir + "/data_stats.pth"

car_images = []
car_actions = []
car_states = []
car_ids = []

for df in data_files:
    combined_data_path = f"{data_dir}/{df}/all_data.pth"
    if os.path.isfile(combined_data_path):
        print(f"[loading data shard: {combined_data_path}]")
        data = torch.load(combined_data_path)
        car_images += data.get("images")
        car_actions += data.get("actions")
        car_states += data.get("states")

car_num = len(car_states)
# car_num = 5

T = 1
# TP = 20
states, actions, next_states, terminals = [], [], [], []
for car_idx in train_car_idx:
    episode_length = min(car_states[car_idx].size(0), car_images[car_idx].size(0))
    if episode_length >= T + 1:
        for t in range(0, episode_length - T):
            actions.append(car_actions[car_idx][t + 1 : t + T + 1])
            states.append(car_states[car_idx][t : t + T].reshape(T, -1))
            next_states.append(car_states[car_idx][t + 1 : t + 1 + T].reshape(T, -1))
            terminals.append(False)
        terminals[-1] = True

states = torch.stack(states)
next_states = torch.stack(next_states)
actions = torch.stack(actions)
terminals = np.array(terminals)

print(f"[loading data stats: {stats_path}]")
stats = torch.load(stats_path)
a_mean = stats.get("a_mean")
a_std = stats.get("a_std")
a_scale = stats.get("a_scale")
# ego vehicle and 6 neighbor vehicles
s_mean = stats.get("s_mean").repeat(7)
s_std = stats.get("s_std").repeat(7)

# Normalise actions, state_vectors
# actions -= a_mean.view(1, 1, 2).expand(actions.size())
# actions /= (1e-8 + a_std.view(1, 1, 2).expand(actions.size()))
actions /= a_scale.view(1, 1, 2).expand(actions.size())

shape = (
    (1, 1, 28) if states.dim() == 3 else (1, 28)
)  # dim = 3: state sequence, dim = 2: single state
states -= s_mean.view(*shape).expand(states.size())
states /= 1e-8 + s_std.view(*shape).expand(states.size())
states = states.reshape((states.size(0), -1))
next_states -= s_mean.view(*shape).expand(next_states.size())
next_states /= 1e-8 + s_std.view(*shape).expand(next_states.size())
next_states = next_states.reshape((next_states.size(0), -1))

actions = actions[:, -1]

print(states.shape[0])

with open(f"../ppuu_expert_{T}_train.pkl", "wb") as f:
    pickle.dump(
        [
            {
                "observations": states.numpy(),
                "next_observations": next_states.numpy(),
                "actions": actions.numpy(),
                "terminals": terminals,
            }
        ],
        f,
    )
