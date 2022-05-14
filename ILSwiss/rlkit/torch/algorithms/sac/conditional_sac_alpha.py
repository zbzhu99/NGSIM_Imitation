from collections import OrderedDict

import numpy as np
import torch

import rlkit.torch.utils.pytorch_util as ptu
from rlkit.torch.algorithms.sac.sac_alpha import SoftActorCritic
from rlkit.core.eval_util import create_stats_ordered_dict


class ConditionalSoftActorCritic(SoftActorCritic):
    """
    version that:
        - uses reparameterization trick
        - has two Q functions
        - has auto-tuned alpha
    """

    def __init__(
        self,
        policy,
        qf1,
        qf2,
        vf,
        **kwargs,
    ):
        super().__init__(
            policy,
            qf1,
            qf2,
            vf,
            **kwargs,
        )

    def train_step(self, batch):
        # q_params = itertools.chain(self.qf1.parameters(), self.qf2.parameters())
        # v_params = itertools.chain(self.vf.parameters())
        # policy_params = itertools.chain(self.policy.parameters())

        rewards = self.reward_scale * batch["rewards"]
        terminals = batch["terminals"]
        obs = batch["observations"]
        latents = batch["latents"]
        actions = batch["actions"]
        next_obs = batch["next_observations"]
        next_latents = batch["latents"]

        """
        QF Loss
        """
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        q1_pred = self.qf1(obs, actions, latent_variable=latents)
        q2_pred = self.qf2(obs, actions, latent_variable=latents)

        # Make sure policy accounts for squashing functions like tanh correctly!
        next_policy_outputs = self.policy(
            next_obs, latent_variable=next_latents, return_log_prob=True
        )
        # in this part, we only need new_actions and log_pi with no grad
        (
            next_new_actions,
            next_policy_mean,
            next_policy_log_std,
            next_log_pi,
        ) = next_policy_outputs[:4]
        target_qf1_values = self.target_qf1(
            next_obs, next_new_actions, latent_variable=next_latents
        )  # do not need grad || it's the shared part of two calculation
        target_qf2_values = self.target_qf2(
            next_obs, next_new_actions, latent_variable=next_latents
        )  # do not need grad || it's the shared part of two calculation
        min_target_value = torch.min(target_qf1_values, target_qf2_values)
        q_target = rewards + (1.0 - terminals) * self.discount * (
            min_target_value - self.alpha * self.reward_scale * next_log_pi
        )  # original implementation has detach
        q_target = q_target.detach()

        qf1_loss = 0.5 * torch.mean((q1_pred - q_target) ** 2)
        qf2_loss = 0.5 * torch.mean((q2_pred - q_target) ** 2)

        # freeze parameter of Q
        # for p in itertools.chain(self.qf1.parameters(), self.qf2.parameters()):
        #     p.requires_grad = False

        qf1_loss.backward()
        qf2_loss.backward()

        self.qf1_optimizer.step()
        self.qf2_optimizer.step()

        """
        Policy Loss
        """

        policy_outputs = self.policy(obs, latent_variable=latents, return_log_prob=True)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]
        q1_new_acts = self.qf1(obs, new_actions, latent_variable=latents)
        q2_new_acts = self.qf2(obs, new_actions, latent_variable=latents)  ## error
        q_new_actions = torch.min(q1_new_acts, q2_new_acts)

        self.policy_optimizer.zero_grad()
        policy_loss = torch.mean(
            self.alpha * self.reward_scale * log_pi - q_new_actions
        )  ##
        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()
        policy_reg_loss = mean_reg_loss + std_reg_loss
        policy_loss = policy_loss + policy_reg_loss
        policy_loss.backward()
        self.policy_optimizer.step()

        """
        Update alpha
        """
        if self.train_alpha:
            log_prob = log_pi.detach() + self.target_entropy
            alpha_loss = -(self.log_alpha * log_prob).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.detach().exp()

        """
        Update networks
        """

        self._update_target_network()

        """
        Save some statistics for eval
        """
        if self.eval_statistics is None:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics = OrderedDict()
            self.eval_statistics["Reward Scale"] = self.reward_scale
            self.eval_statistics["QF1 Loss"] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics["QF2 Loss"] = np.mean(ptu.get_numpy(qf2_loss))
            if self.train_alpha:
                self.eval_statistics["Alpha Loss"] = np.mean(ptu.get_numpy(alpha_loss))
            self.eval_statistics["Policy Loss"] = np.mean(ptu.get_numpy(policy_loss))
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Q1 Predictions",
                    ptu.get_numpy(q1_pred),
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Q2 Predictions",
                    ptu.get_numpy(q2_pred),
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Alpha",
                    [ptu.get_numpy(self.alpha)],
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Log Pis",
                    ptu.get_numpy(log_pi),
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Policy mu",
                    ptu.get_numpy(policy_mean),
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Policy log std",
                    ptu.get_numpy(policy_log_std),
                )
            )
