import numpy as np
import pickle
from sklearn.neighbors._kde import KernelDensity
import os
import sys
import joblib
import torch
import json
from tqdm import tqdm

sys.path.append("/NAS2020/Workspaces/DRLGroup/zbzhu/lfo-ppuu/lfo")

from dataloader import DataLoader
from map_i80_ctrl import ControlledI80
from tianshou.env import SubprocVectorEnv
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.data_management.split_dict import split_dict


s_std = np.tile(np.array([392.1703, 44.0625, 24.4669, 1.0952]), 7)
s_mean = np.tile(np.array([887.6, 117.67, 36.453, -0.23616]), 7)


def kl_divergence(x1, x2):
    p = kde_prob(x1, min_v=0, max_v=1, scale=100)
    q = kde_prob(x2, min_v=0, max_v=1, scale=100)
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def kde_prob(x, min_v=0, max_v=1, scale=100):
    kde = KernelDensity(kernel="gaussian", bandwidth=(max_v - min_v) * 1.0 / scale).fit(
        list(x)
    )  # x.shape: [None, 2]
    data = [
        (i * 1.0 / scale, j * 1.0 / scale)
        for i in range(min_v * scale, max_v * scale)
        for j in range(min_v * scale, max_v * scale)
    ]
    prob = np.exp(kde.score_samples(data)) + 1e-4  # x.shape: [None, 1]
    return prob


def obs_unnorm(obs):
    obs *= s_std
    obs += s_mean
    return obs


def make_env(env_kwargs, rank, seed=0, car_index=None):
    def _init():
        """
        env_specs:
            env_name: 'halfcheetah'
            env_kwargs: {} # kwargs to pass to the env constructor call
        """
        env = ControlledI80(**env_kwargs)
        env.seed(rank + seed)
        if car_index is not None and hasattr(env, "set_train_indx"):
            env.set_train_indx(car_index)
        return env

    return _init


class opt:
    debug = 0


demo_path = "/NAS2020/Workspaces/DRLGroup/zbzhu/lfo-ppuu/expert_demo_xy.pkl"
test_idx_path = "/NAS2020/Workspaces/DRLGroup/zbzhu/lfo-ppuu/lfo/demos/expert_trajs_50/PPUU/test_indx_final.pkl"
log_path = "/NAS2020/Workspaces/DRLGroup/zbzhu/lfo-ppuu/lfo/logs/gailfo-ppuu-final/gailfo_ppuu_final--2021_01_26_03_20_45--s-0"
model_path = os.path.join(log_path, "best.pkl")
variant_path = os.path.join(log_path, "variant.json")

with open(variant_path, "rb") as f:
    variant = json.load(f)

env_kwargs = dict(
    fps=30,
    nb_states=1,
    display=False,
    delta_t=0.1,
    store=False,
    show_frame_count=False,
    data_dir="ppuu_logs/",
)


if __name__ == "__main__":
    env_num = 50
    env_wait_num = 25

    with open(test_idx_path, "rb") as f:
        test_idx = pickle.load(f)

    splited_eval_dict = split_dict(test_idx, env_num)
    eval_car_num = [len(d) for d in splited_eval_dict]

    envs = SubprocVectorEnv(
        [
            make_env(
                env_kwargs,
                i,
                car_index=splited_eval_dict[i],
            )
            for i in range(env_num)
        ],
        wait_num=env_wait_num,
    )

    if os.path.isfile(demo_path):
        with open(demo_path, "rb") as f:
            all_demo_x, all_demo_y = pickle.load(f)
    else:
        dataloader = DataLoader(None, opt, "i80")
        all_demo_x, all_demo_y = [], []
        for idx in test_idx.keys():
            all_demo_x.extend(dataloader.states[idx][:, 0, 0].numpy())
            all_demo_y.extend(dataloader.states[idx][:, 0, 1].numpy())
        with open(demo_path, "wb") as f:
            pickle.dump((all_demo_x, all_demo_y), f)

    model = joblib.load(model_path)
    policy = model["policy"]
    eval_policy = MakeDeterministic(policy)

    all_agent_x, all_agent_y = [], []
    items = list(test_idx.items())

    ready_env_ids = np.arange(env_num)
    finished_env_ids = []

    obs_list = envs.reset()
    done = False
    episode_step = np.zeros(env_num)
    env_finished_car_num = np.zeros(env_num)

    pbar = tqdm(total=len(items))
    while True:
        actions = []
        for obs in obs_list[ready_env_ids]:
            ori_obs = obs_unnorm(obs.copy())
            agent_x = ori_obs[0]
            agent_y = ori_obs[1]
            all_agent_x.append(agent_x)
            all_agent_y.append(agent_y)
            with torch.no_grad():
                action, _ = eval_policy.get_action(obs_np=obs)
            actions.append(action)
        actions = np.array(actions)

        next_obs_list, rews, dones, env_infos = envs.step(actions, id=ready_env_ids)

        ready_env_ids = np.array([i["env_id"] for i in env_infos])

        obs_list[ready_env_ids] = next_obs_list

        for idx, done in enumerate(dones):
            env_id = ready_env_ids[idx]
            episode_step[env_id] += 1
            if done or episode_step[env_id] > 1500:
                env_finished_car_num[env_id] += 1
                pbar.update(1)
                if not done:
                    obs_list[env_id] = envs.reset(id=env_id)
                if env_finished_car_num[env_id] == eval_car_num[env_id]:
                    finished_env_ids.append(env_id)

        ready_env_ids = np.array(
            [x for x in ready_env_ids if x not in finished_env_ids]
        )

        if len(finished_env_ids) == env_num:
            assert len(ready_env_ids) == 0
            break
    pbar.close()

    all_agent_x = np.array(all_agent_x)[:, np.newaxis] / 1600
    all_agent_y = np.array(all_agent_y)[:, np.newaxis] / 200
    all_agent_pos = np.concatenate((all_agent_x, all_agent_y), 1)

    all_demo_x = np.array(all_demo_x)[:, np.newaxis] / 1600
    all_demo_y = np.array(all_demo_y)[:, np.newaxis] / 200
    all_demo_pos = np.concatenate((all_demo_x, all_demo_y), 1)

    kld = kl_divergence(all_agent_pos, all_demo_pos)
    print(kld)
