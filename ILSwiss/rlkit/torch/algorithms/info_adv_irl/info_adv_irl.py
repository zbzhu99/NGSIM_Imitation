import numpy as np
from collections import OrderedDict, defaultdict

import torch
import torch.optim as optim
import torch.nn.functional as F

import rlkit.torch.utils.pytorch_util as ptu
from rlkit.torch.algorithms.adv_irl.adv_irl import AdvIRL
from rlkit.data_management.path_builder import PathBuilder
import gtimer as gt
from tqdm import tqdm
from typing import Dict, List
from rlkit.core import logger, eval_util, dict_list_to_list_dict


class InfoAdvIRL(AdvIRL):
    """
    Depending on choice of reward function and size of replay
    buffer this will be:
        - AIRL
        - GAIL (without extra entropy term)
        - FAIRL
        - Discriminator Actor Critic

    I did not implement the reward-wrapping mentioned in
    https://arxiv.org/pdf/1809.02925.pdf though

    Features removed from v1.0:
        - gradient clipping
        - target disc (exponential moving average disc)
        - target policy (exponential moving average policy)
        - disc input noise
    """

    def __init__(
        self,
        mode,  # airl, gail, or fairl
        discriminator_n,
        policy_trainer_n,
        posterior_trainer_n,
        expert_replay_buffer,
        state_only=False,
        disc_optim_batch_size=1024,
        policy_optim_batch_size=1024,
        posterior_optim_batch_size=1024,
        policy_optim_batch_size_from_expert=0,
        num_update_loops_per_train_call=1,
        num_disc_updates_per_loop_iter=100,
        num_policy_updates_per_loop_iter=100,
        num_post_updates_per_loop_iter=100,
        disc_lr=1e-3,
        disc_focal_loss_gamma=0.0,
        disc_momentum=0.9,
        disc_optimizer_class=optim.Adam,
        post_r_coef=0.1,
        latent_distribution=None,
        use_grad_pen=True,
        grad_pen_weight=10,
        rew_clip_min=None,
        rew_clip_max=None,
        **kwargs,
    ):
        super().__init__(
            mode,
            discriminator_n,
            policy_trainer_n,
            expert_replay_buffer,
            state_only=state_only,
            disc_optim_batch_size=disc_optim_batch_size,
            policy_optim_batch_size=policy_optim_batch_size,
            policy_optim_batch_size_from_expert=policy_optim_batch_size_from_expert,
            num_update_loops_per_train_call=num_update_loops_per_train_call,
            num_disc_updates_per_loop_iter=num_disc_updates_per_loop_iter,
            num_policy_updates_per_loop_iter=num_policy_updates_per_loop_iter,
            disc_lr=disc_lr,
            disc_focal_loss_gamma=disc_focal_loss_gamma,
            disc_momentum=disc_momentum,
            disc_optimizer_class=disc_optimizer_class,
            use_grad_pen=use_grad_pen,
            grad_pen_weight=grad_pen_weight,
            rew_clip_min=rew_clip_min,
            rew_clip_max=rew_clip_max,
            **kwargs,
        )

        self.eval_sampler._init_latent_distribution(latent_distribution)
        self.post_r_coef = post_r_coef
        self.posterior_trainer_n = posterior_trainer_n
        self.posterior_optim_batch_size = posterior_optim_batch_size
        self.num_post_updates_per_loop_iter = num_post_updates_per_loop_iter

        self.latent_distribution = latent_distribution
        self.latents_n = np.array(
            [
                {a_id: self._get_random_latent_variable() for a_id in self.agent_ids}
                for _ in range(self.training_env_num)
            ]
        )

    def _get_random_latent_variable(self):
        latent = (
            self.latent_distribution.sample_prior(batch_size=1)
            .cpu()
            .numpy()
            .reshape(-1)
        )
        return latent

    def _end_epoch(self):
        for p_id in self.policy_ids:
            self.posterior_trainer_n[p_id].end_epoch()
        super()._end_epoch()

    def _do_training(self, epoch):
        for t in range(self.num_update_loops_per_train_call):
            for a_id in self.agent_ids:
                for _ in range(self.num_disc_updates_per_loop_iter):
                    self._do_reward_training(epoch, a_id)
                for _ in range(self.num_post_updates_per_loop_iter):
                    self._do_posterior_training(epoch, a_id)
                for _ in range(self.num_policy_updates_per_loop_iter):
                    self._do_policy_training(epoch, a_id)

    def _do_policy_training(self, epoch, agent_id):

        policy_id = self.policy_mapping_dict[agent_id]

        if self.policy_optim_batch_size_from_expert > 0:
            policy_batch_from_policy_buffer = self.get_batch(
                self.policy_optim_batch_size - self.policy_optim_batch_size_from_expert,
                agent_id,
                False,
            )
            policy_batch_from_expert_buffer = self.get_batch(
                self.policy_optim_batch_size_from_expert,
                agent_id,
                True,
            )
            policy_batch = {}
            for k in policy_batch_from_policy_buffer:
                policy_batch[k] = torch.cat(
                    [
                        policy_batch_from_policy_buffer[k],
                        policy_batch_from_expert_buffer[k],
                    ],
                    dim=0,
                )
        else:
            policy_batch = self.get_batch(self.policy_optim_batch_size, agent_id, False)

        obs = policy_batch["observations"]
        acts = policy_batch["actions"]
        next_obs = policy_batch["next_observations"]
        latents = policy_batch["latents"]

        self.discriminator_n[policy_id].eval()
        if self.state_only:
            disc_input = torch.cat([obs, next_obs], dim=1)
        else:
            disc_input = torch.cat([obs, acts], dim=1)
        disc_logits = self.discriminator_n[policy_id](disc_input).detach()
        self.discriminator_n[policy_id].train()

        # compute the reward using the algorithm
        if self.mode == "airl":
            # If you compute log(D) - log(1-D) then you just get the logits
            policy_batch["rewards"] = disc_logits
        elif self.mode == "gail":  # -log (1-D) > 0
            policy_batch["rewards"] = F.softplus(
                disc_logits, beta=1
            )  # F.softplus(disc_logits, beta=-1)
        elif self.mode == "gail2":  # log D < 0
            policy_batch["rewards"] = F.softplus(
                disc_logits, beta=-1
            )  # F.softplus(disc_logits, beta=-1)
        elif self.mode == "gail3":  # log D < 0
            origin_reward = F.softplus(disc_logits, beta=-1)
            policy_batch["rewards"] = torch.clamp(origin_reward + 2.5, min=0, max=2.5)
        else:  # fairl
            policy_batch["rewards"] = torch.exp(disc_logits) * (-1.0 * disc_logits)

        # info-GAIL reward
        if self.mode == "gail":
            # positive reward
            posterior = self.posterior_trainer_n[
                policy_id
            ].target_posterior_model.get_posterior(obs, acts, latents)

            def arc_sigmoid(x):
                x = torch.clamp(x, min=0.01, max=0.99)
                x = torch.log(x / (1.0 - x))
                return x

            positive_r = F.softplus(arc_sigmoid(posterior), beta=1)
            policy_batch["rewards"] += self.post_r_coef * positive_r

        elif self.mode == "gail2":
            # negative reward, same as paper.
            log_posterior = self.posterior_trainer_n[
                policy_id
            ].target_posterior_model.get_log_posterior(obs, acts, latents)
            assert (
                log_posterior.shape == policy_batch["rewards"].shape
            ), "{}, {}".format(log_posterior.shape, policy_batch["rewards"].shape)
            policy_batch["rewards"] += self.post_r_coef * log_posterior

        elif self.mode == "gail3":  # log D < 0
            log_posterior = self.posterior_trainer_n[
                policy_id
            ].target_posterior_model.get_log_posterior(obs, acts, latents)
            policy_batch["rewards"] += self.post_r_coef * torch.clamp(
                log_posterior + 2.5, min=0, max=2.5
            )

        if self.clip_max_rews:
            policy_batch["rewards"] = torch.clamp(
                policy_batch["rewards"], max=self.rew_clip_max
            )
        if self.clip_min_rews:
            policy_batch["rewards"] = torch.clamp(
                policy_batch["rewards"], min=self.rew_clip_min
            )

        # policy optimization step
        self.policy_trainer_n[policy_id].train_step(policy_batch)

        self.disc_eval_statistics[f"{agent_id} Disc Rew Mean"] = np.mean(
            ptu.get_numpy(policy_batch["rewards"])
        )
        self.disc_eval_statistics[f"{agent_id} Disc Rew Std"] = np.std(
            ptu.get_numpy(policy_batch["rewards"])
        )
        self.disc_eval_statistics[f"{agent_id} Disc Rew Max"] = np.max(
            ptu.get_numpy(policy_batch["rewards"])
        )
        self.disc_eval_statistics[f"{agent_id} Disc Rew Min"] = np.min(
            ptu.get_numpy(policy_batch["rewards"])
        )

    def _do_posterior_training(self, epoch, agent_id):
        """
        Train the Posterior network
        """
        policy_id = self.policy_mapping_dict[agent_id]
        batch = self.get_batch(self.posterior_optim_batch_size, agent_id, False)
        # print(f"batch.keys: {batch.keys()}")
        self.posterior_trainer_n[policy_id].train_step(batch)

    @property
    def networks_n(self):
        return {
            p_id: [self.discriminator_n[p_id]] + self.policy_trainer_n[p_id].networks
            for p_id in self.policy_ids
        }

    def get_epoch_snapshot(self, epoch):
        # snapshot = super().get_epoch_snapshot(epoch)
        snapshot = dict(epoch=epoch)
        for p_id in self.policy_ids:
            snapshot[p_id] = self.policy_trainer_n[p_id].get_snapshot()
            snapshot[p_id].update(self.posterior_trainer_n[p_id].get_snapshot())
            snapshot[p_id].update(disc=self.discriminator_n[p_id])
        return snapshot

    def start_training(self, start_epoch=0):
        self._start_new_rollout()

        self._current_path_builder = [
            PathBuilder(
                self.agent_ids,
                self.training_env.sub_envs_info[env_id].scenario_name,
                self.training_env.sub_envs_info[env_id].traffic_name,
            )
            for env_id in range(self.training_env_num)
        ]

        for epoch in gt.timed_for(
            range(start_epoch, self.num_epochs),
            save_itrs=True,
        ):
            self._start_epoch(epoch)
            for _ in tqdm(
                range(self.num_env_steps_per_epoch // self.training_env_wait_num),
                unit_scale=self.training_env_wait_num,
            ):
                self.actions_n[self._ready_env_ids] = self._get_action_and_info(
                    self.observations_n[self._ready_env_ids],
                    self.latents_n[self._ready_env_ids],
                )

                for action_n in self.actions_n:
                    for a_id, action in action_n.items():
                        if type(action) is tuple:
                            action_n[a_id] = action_n[a_id][0]

                if self.render:
                    self.training_env.render()

                (
                    next_obs_n,
                    rewards_n,
                    terminals_n,
                    env_infos_n,
                ) = self.training_env.step(
                    self.actions_n[self._ready_env_ids], self._ready_env_ids
                )
                self._ready_env_ids = np.array([i["env_id"] for i in env_infos_n])

                if self.no_terminal:
                    terminals_n = [
                        dict(
                            zip(
                                terminal_n.keys(),
                                [False for _ in range(len(terminal_n))],
                            )
                        )
                        for terminal_n in terminals_n
                    ]
                self._n_env_steps_total += self.training_env_wait_num

                self._handle_vec_step(
                    self.observations_n[self._ready_env_ids],
                    self.actions_n[self._ready_env_ids],
                    rewards_n,
                    self.latents_n[self._ready_env_ids],
                    next_obs_n,
                    terminals_n,
                    env_ids=self._ready_env_ids,
                    env_infos_n=env_infos_n,
                )

                terminals_all = [
                    np.all(list(terminal.values())) for terminal in terminals_n
                ]

                self.observations_n[self._ready_env_ids] = next_obs_n

                if np.any(terminals_all):
                    end_env_id = self._ready_env_ids[np.where(terminals_all)[0]]
                    self._handle_vec_rollout_ending(end_env_id)
                    if not self.training_env.auto_reset:
                        self.observations_n[end_env_id] = self.training_env.reset(
                            end_env_id
                        )
                elif np.any(
                    np.array(
                        [
                            len(self._current_path_builder[i])
                            for i in self._ready_env_ids
                        ]
                    )
                    >= self.max_path_length
                ):

                    env_ind_local = [
                        i
                        for i in self._ready_env_ids
                        if len(self._current_path_builder[i]) >= self.max_path_length
                    ]
                    self._handle_vec_rollout_ending(env_ind_local)
                    self.observations_n[env_ind_local] = self.training_env.reset(
                        env_ind_local
                    )

                if (
                    self._n_env_steps_total - self._n_prev_train_env_steps
                ) >= self.num_steps_between_train_calls:
                    gt.stamp("sample")
                    self._try_to_train(epoch)
                    gt.stamp("train")

            gt.stamp("sample")
            self._try_to_eval(epoch)
            gt.stamp("eval")
            self._end_epoch()

    def _get_action_and_info(
        self, observations_n: List[Dict[str, np.ndarray]], latents_n
    ):
        """
        Get an action to take in the environment.
        :param observation_n:
        :return:
        """
        actions_n = [{} for _ in range(len(observations_n))]
        policy_obs_list_n = defaultdict(list)  # to recover the corresponding actions
        policy_latent_list_n = defaultdict(list)  # to recover the corresponding actions
        policy_to_env_agent_id = defaultdict(list)
        for idx, observation_n in enumerate(observations_n):  # for each env
            latent_n = latents_n[idx]
            for agent_id in observation_n.keys():  # for each agent in single env.
                policy_id = self.policy_mapping_dict[agent_id]
                self.exploration_policy_n[policy_id].set_num_steps_total(
                    self._n_env_steps_total
                )
                policy_obs_list_n[policy_id].append(observation_n[agent_id])
                policy_latent_list_n[policy_id].append(latent_n[agent_id])
                policy_to_env_agent_id[policy_id].append((idx, agent_id))
        for policy_id in policy_obs_list_n:
            stacked_obs = np.stack(policy_obs_list_n[policy_id])
            stacked_latent = np.stack(policy_latent_list_n[policy_id])
            actions = self.exploration_policy_n[policy_id].get_actions(
                stacked_obs, stacked_latent
            )
            for i in range(actions.shape[0]):
                idx, agent_id = policy_to_env_agent_id[policy_id][i]
                actions_n[idx][agent_id] = actions[i]
        return actions_n

    def _handle_vec_step(
        self,
        observations_n: List,
        actions_n: List,
        rewards_n: List,
        latents_n: List,
        next_observations_n: List,
        terminals_n: List,
        env_infos_n: List,
        env_ids: List,
    ):
        """
        Implement anything that needs to happen after every step under vec envs
        :return:
        """
        for (
            ob_n,
            action_n,
            reward_n,
            latent_n,
            next_ob_n,
            terminal_n,
            env_info_n,
            env_id,
        ) in zip(
            observations_n,
            actions_n,
            rewards_n,
            latents_n,
            next_observations_n,
            terminals_n,
            env_infos_n,
            env_ids,
        ):
            self._handle_step(
                ob_n,
                action_n,
                reward_n,
                latent_n,
                next_ob_n,
                terminal_n,
                env_info_n=env_info_n,
                env_id=env_id,
                add_buf=False,
            )

    def _handle_step(
        self,
        observation_n,
        action_n,
        reward_n,
        latent_n,
        next_observation_n,
        terminal_n,
        env_info_n,
        env_id=None,
        add_buf=True,
        path_builder=True,
    ):
        """
        Implement anything that needs to happen after every step
        :return:
        """
        if path_builder:
            assert env_id is not None
            for a_id in observation_n.keys():
                # some agents may terminate earlier than others
                if a_id not in next_observation_n.keys():
                    continue
                self._current_path_builder[env_id][a_id].add_all(
                    observations=observation_n[a_id],
                    actions=action_n[a_id],
                    rewards=reward_n[a_id],
                    latents=latent_n[a_id],
                    next_observations=next_observation_n[a_id],
                    terminals=terminal_n[a_id],
                    env_infos=env_info_n[a_id],
                )

        if add_buf:
            self.replay_buffer.add_sample(
                observation_n=observation_n,
                action_n=action_n,
                reward_n=reward_n,
                terminal_n=terminal_n,
                latent_n=latent_n,
                next_observation_n=next_observation_n,
                env_info_n=env_info_n,
            )

    def _handle_vec_rollout_ending(self, end_idx):
        """
        Implement anything that needs to happen after every vec env rollout.
        """
        super()._handle_vec_rollout_ending(end_idx)
        for env_id in end_idx:
            for a_id in self.agent_ids:
                self.latents_n[env_id][a_id] = self._get_random_latent_variable()

    def _handle_path(self, path, env_id=None):
        """
        Naive implementation: just loop through each transition.
        :param path:
        :return:
        """
        for (
            ob_n,
            action_n,
            reward_n,
            latent_n,
            next_ob_n,
            terminal_n,
            env_info_n,
        ) in zip(
            *map(
                dict_list_to_list_dict,
                [
                    path.get_all_agent_dict("observations"),
                    path.get_all_agent_dict("actions"),
                    path.get_all_agent_dict("rewards"),
                    path.get_all_agent_dict("latents"),
                    path.get_all_agent_dict("next_observations"),
                    path.get_all_agent_dict("terminals"),
                    path.get_all_agent_dict("env_infos"),
                ],
            )
        ):
            self._handle_step(
                ob_n,
                action_n,
                reward_n,
                latent_n,
                next_ob_n,
                terminal_n,
                env_info_n=env_info_n,
                path_builder=False,
                env_id=env_id,
            )

    @property
    def networks_n(self):
        return {
            p_id: [self.discriminator_n[p_id]]
            + self.policy_trainer_n[p_id].networks
            + self.posterior_trainer_n[p_id].networks
            for p_id in self.policy_ids
        }

    def evaluate(self, epoch):

        # InfoAdvIRL evaluate
        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()
        for p_id in self.policy_ids:
            posterior_statistics = self.posterior_trainer_n[p_id].get_eval_statistics()
            for name, data in posterior_statistics.items():
                self.eval_statistics.update({f"{p_id} {name}": data})

        # AdvIRL evaluate
        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()
        self.eval_statistics.update(self.disc_eval_statistics)
        for p_id in self.policy_ids:
            _statistics = self.policy_trainer_n[p_id].get_eval_statistics()
            for name, data in _statistics.items():
                self.eval_statistics.update({f"{p_id} {name}": data})

        # BaseAlgorithm evaluate
        statistics = OrderedDict()
        try:
            statistics.update(self.eval_statistics)
            self.eval_statistics = None
        except Exception:
            print("No Stats to Eval")

        logger.log("Collecting samples for evaluation")
        test_paths = self.eval_sampler.obtain_samples()

        statistics.update(
            eval_util.get_generic_path_information(
                test_paths,
                self.env,
                stat_prefix="Test",
                # scenario_stats_class=eval_util.InfoAdvIRLScenarioWiseStats,
            )
        )

        if len(self._exploration_paths) > 0:
            statistics.update(
                eval_util.get_generic_path_information(
                    self._exploration_paths,
                    self.env,
                    stat_prefix="Exploration",
                    # scenario_stats_class=eval_util.InfoAdvIRLScenarioWiseStats,
                )
            )

        if hasattr(self.env, "log_diagnostics"):
            self.env.log_diagnostics(test_paths)
        if hasattr(self.env, "log_statistics"):
            statistics.update(self.env.log_statistics(test_paths))
        if int(epoch) % self.freq_log_visuals == 0:
            if hasattr(self.env, "log_visuals"):
                self.env.log_visuals(test_paths, epoch, logger.get_snapshot_dir())

        agent_mean_avg_returns = eval_util.get_agent_mean_avg_returns(test_paths)
        statistics["AgentMeanAverageReturn"] = agent_mean_avg_returns
        for key, value in statistics.items():
            logger.record_tabular(key, np.mean(value))

        best_statistic = statistics[self.best_key]
        data_to_save = {"epoch": epoch, "statistics": statistics}
        data_to_save.update(self.get_epoch_snapshot(epoch))
        if self.save_epoch:
            logger.save_extra_data(data_to_save, "epoch{}.pkl".format(epoch))
            print("\n\nSAVED MODEL AT EPOCH {}\n\n".format(epoch))

        if self.best_criterion == "largest":
            if best_statistic > self.best_statistic_so_far:
                self.best_statistic_so_far = best_statistic
                if self.save_best and epoch >= self.save_best_starting_from_epoch:
                    data_to_save = {"epoch": epoch, "statistics": statistics}
                    data_to_save.update(self.get_epoch_snapshot(epoch))
                    logger.save_extra_data(data_to_save, "best.pkl")
                    print("\n\nSAVED BEST\n\n")
        elif self.best_criterion == "smallest":
            if best_statistic < self.best_statistic_so_far:
                self.best_statistic_so_far = best_statistic
                if self.save_best and epoch >= self.save_best_starting_from_epoch:
                    data_to_save = {"epoch": epoch, "statistics": statistics}
                    data_to_save.update(self.get_epoch_snapshot(epoch))
                    logger.save_extra_data(data_to_save, "best.pkl")
                    print("\n\nSAVED BEST\n\n")
        else:
            raise NotImplementedError
