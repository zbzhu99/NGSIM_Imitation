import math
import torch
from torch import nn as nn
import torch.nn.functional as F

from rlkit.policies.base import ExplorationPolicy, Policy
from rlkit.torch.common.networks import Mlp, ConditionalMlp
from rlkit.torch.common.distributions import ReparamTanhMultivariateNormal
from rlkit.torch.common.distributions import ReparamMultivariateNormalDiag
from rlkit.torch.core import PyTorchModule

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class MakeDeterministic(Policy):
    def __init__(self, stochastic_policy):
        self.stochastic_policy = stochastic_policy

    def get_action(self, *args, **kwargs):
        return self.stochastic_policy.get_action(deterministic=True, *args, **kwargs)

    def get_actions(self, *args, **kwargs):
        return self.stochastic_policy.get_actions(deterministic=True, *args, **kwargs)

    def get_options(self, *args, **kwargs):
        return self.stochastic_policy.get_options(deterministic=True, *args, **kwargs)

    def train(self, mode):
        pass

    def set_num_steps_total(self, num):
        pass

    def to(self, device):
        self.stochastic_policy.to(device)


class DiscretePolicy(Mlp, ExplorationPolicy):
    """
    Usage:

    ```
    policy = DiscretePolicy(...)
    action, log_prob = policy(obs, return_log_prob=True)
    ```
    """

    def __init__(self, hidden_sizes, obs_dim, action_dim, init_w=1e-3, **kwargs):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            output_activation=nn.LogSoftmax(1),
            **kwargs,
        )

    def get_action(self, obs_np, deterministic=False):
        action = self.get_actions(obs_np[None], deterministic=deterministic)
        return action[0], {}

    def get_actions(self, obs_np, deterministic=False):
        return self.eval_np(obs_np, deterministic=deterministic)[0]

    def forward(
        self,
        obs,
        deterministic=False,
        return_log_prob=False,
    ):
        log_probs, pre_act = super().forward(obs, return_preactivations=True)

        if deterministic:
            log_prob, idx = torch.max(log_probs, 1)
            return (idx, None)
        else:
            # Using Gumbel-Max trick to sample from the multinomials
            u = torch.rand(pre_act.size(), requires_grad=False, device=pre_act.device)
            gumbel = -torch.log(-torch.log(u))
            _, idx = torch.max(gumbel + pre_act, 1)

            log_prob = torch.gather(log_probs, 1, idx.unsqueeze(1))

            # # print(log_probs.size(-1))
            # # print(log_probs.data.numpy())
            # # print(np.exp(log_probs.data.numpy()))
            # idx = choice(
            #     log_probs.size(-1),
            #     size=1,
            #     p=np.exp(log_probs.data.numpy())
            # )
            # log_prob = log_probs[0,idx]

            # print(idx)
            # print(log_prob)

            return (idx, log_prob)

    def get_log_pis(self, obs):
        return super().forward(obs)


class MlpPolicy(Mlp, ExplorationPolicy):
    def __init__(self, hidden_sizes, obs_dim, action_dim, init_w=1e-3, **kwargs):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs,
        )

    def get_action(self, obs_np, deterministic=False):
        """
        deterministic=False makes no diff, just doing this for
        consistency in interface for now
        """
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        actions = actions[None]
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return self.eval_np(obs_np)[0]


class MlpGaussianNoisePolicy(Mlp, ExplorationPolicy):
    def __init__(
        self,
        hidden_sizes,
        obs_dim,
        action_dim,
        init_w=1e-3,
        policy_noise=0.1,
        policy_noise_clip=0.5,
        max_act=1.0,
        **kwargs,
    ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs,
        )
        self.noise = policy_noise
        self.noise_clip = policy_noise_clip
        self.max_act = max_act

    def get_action(self, obs_np, deterministic=False):
        """
        deterministic=False makes no diff, just doing this for
        consistency in interface for now
        """
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        actions = actions[None]
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return self.eval_np(obs_np, deterministic=deterministic)[0]

    def forward(self, obs, deterministic=False):
        h = obs
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm:
                h = self.layer_norms[i](h)
            if self.batch_norm:
                h = self.batch_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        if self.batch_norm_before_output_activation:
            preactivation = self.batch_norms[-1](preactivation)
        action = self.max_act * self.output_activation(preactivation)
        if deterministic:
            pass
        else:
            noise = self.noise * torch.normal(
                torch.zeros_like(action),
            )
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            action += noise

        return (action, preactivation)


class ReparamTanhMultivariateGaussianPolicy(Mlp, ExplorationPolicy):
    """
    Usage:

    ```
    policy = ReparamTanhMultivariateGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```

    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.

    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """

    def __init__(
        self,
        hidden_sizes,
        obs_dim,
        action_dim,
        init_w=1e-3,
        max_act=1.0,
        conditioned_std: bool = False,
        **kwargs,
    ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs,
        )
        self.max_act = max_act
        self.conditioned_std = conditioned_std

        if self.conditioned_std:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.action_log_std = nn.Parameter(torch.zeros(1, action_dim))

    @torch.jit.ignore
    def get_action(self, obs_np, deterministic=False):
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {}

    @torch.jit.ignore
    def get_actions(self, obs_np, deterministic=False):
        return self.eval_np(obs_np, deterministic=deterministic)[0]

    @torch.jit.ignore
    def forward(
        self, obs, deterministic=False, return_log_prob=False, return_tanh_normal=False
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """

        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))

        mean = self.last_fc(h)

        if self.conditioned_std:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        else:
            log_std = self.action_log_std.expand_as(mean)
        std = torch.exp(log_std)

        log_prob = None
        expected_log_prob = None
        mean_action_log_prob = None
        pre_tanh_value = None
        if deterministic:
            action = torch.tanh(mean)
        else:
            tanh_normal = ReparamTanhMultivariateNormal(mean, log_std)

            if return_log_prob:
                action, pre_tanh_value = tanh_normal.sample(return_pretanh_value=True)
                log_prob = tanh_normal.log_prob(action, pre_tanh_value=pre_tanh_value)
            else:
                action = tanh_normal.sample()

        # doing it like this for now for backwards compatibility
        if return_tanh_normal:
            return (
                action,
                mean,
                log_std,
                log_prob,
                expected_log_prob,
                std,
                mean_action_log_prob,
                pre_tanh_value,
                tanh_normal,
            )
        return (
            action,
            mean,
            log_std,
            log_prob,
            expected_log_prob,
            std,
            mean_action_log_prob,
            pre_tanh_value,
        )

    @torch.jit.export
    def jit_forward(self, obs):
        """torch.jit does not support condition control
        :param obs: Observation
        """
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        action = torch.tanh(mean)
        return action

    @torch.jit.ignore
    def get_log_prob_entropy(self, obs, acts):
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        if self.conditioned_std:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        else:
            log_std = self.action_log_std.expand_as(mean)

        tanh_normal = ReparamTanhMultivariateNormal(mean, log_std)
        log_prob = tanh_normal.log_prob(acts)

        entropy = (0.5 + 0.5 * math.log(2 * math.pi) + log_std).sum(
            dim=-1, keepdim=True
        )

        return log_prob, entropy

    @torch.jit.ignore
    def get_log_prob(self, obs, acts, return_normal_params=False):
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        if self.conditioned_std:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        else:
            log_std = self.action_log_std.expand_as(mean)

        tanh_normal = ReparamTanhMultivariateNormal(mean, log_std)
        log_prob = tanh_normal.log_prob(acts)

        if return_normal_params:
            return log_prob, mean, log_std
        return log_prob


class ConditionalReparamTanhMultivariateGaussianPolicy(
    PyTorchModule, ExplorationPolicy
):
    """
    Usage:

    ```
    policy = ConditionalReparamTanhMultivariateGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs, latent_variable)
    action, mean, log_std, _ = policy(obs, latent_variable, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, latent_variable, return_log_prob=True)
    ```

    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.

    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """

    def __init__(
        self,
        obs_hidden_sizes,
        obs_dim,
        action_dim,
        latent_distribution,
        latent_hidden_sizes,
        init_w=1e-3,
        max_act=1.0,
        conditioned_std: bool = False,
        hidden_activation=F.relu,
        **kwargs,
    ):
        self.save_init_params(locals())
        super().__init__()

        self.latent_input_dim = latent_distribution.dim

        self.mlp = ConditionalMlp(
            input_hidden_sizes=obs_hidden_sizes[:-1],
            input_size=obs_dim,
            output_size=obs_hidden_sizes[-1],
            latent_input_dim=latent_distribution.dim,
            latent_hidden_sizes=latent_hidden_sizes,
            output_activation=hidden_activation,
            **kwargs,
        )

        self.max_act = max_act
        self.conditioned_std = conditioned_std
        self.hidden_activation = hidden_activation
        self.last_hidden_size = self.mlp.output_size

        self.last_fc = nn.Linear(self.last_hidden_size, action_dim)

        if self.conditioned_std:
            self.last_fc_log_std = nn.Linear(self.last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.action_log_std = nn.Parameter(torch.zeros(1, action_dim))

    @torch.jit.ignore
    def get_action(self, obs_np, latent_variable_np, deterministic=False):
        actions = self.get_actions(
            obs_np[None], latent_variable_np[None], deterministic=deterministic
        )
        return actions[0, :], {}

    @torch.jit.ignore
    def get_actions(self, obs_np, latent_variable_np, deterministic=False):
        return self.eval_np(obs_np, latent_variable_np, deterministic=deterministic)[0]

    @torch.jit.ignore
    def forward(
        self,
        obs,
        latent_variable,
        deterministic=False,
        return_log_prob=False,
        return_tanh_normal=False,
    ):
        """
        :param obs: Observation
        :param latent_variable: torch.long
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """

        h = self.mlp(obs, latent_variable)
        mean = self.last_fc(h)

        if self.conditioned_std:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        else:
            log_std = self.action_log_std.expand_as(mean)
        std = torch.exp(log_std)

        log_prob = None
        expected_log_prob = None
        mean_action_log_prob = None
        pre_tanh_value = None
        if deterministic:
            action = torch.tanh(mean)
        else:
            tanh_normal = ReparamTanhMultivariateNormal(mean, log_std)

            if return_log_prob:
                action, pre_tanh_value = tanh_normal.sample(return_pretanh_value=True)
                log_prob = tanh_normal.log_prob(action, pre_tanh_value=pre_tanh_value)
            else:
                action = tanh_normal.sample()

        # doing it like this for now for backwards compatibility
        if return_tanh_normal:
            return (
                action,
                mean,
                log_std,
                log_prob,
                expected_log_prob,
                std,
                mean_action_log_prob,
                pre_tanh_value,
                tanh_normal,
            )
        return (
            action,
            mean,
            log_std,
            log_prob,
            expected_log_prob,
            std,
            mean_action_log_prob,
            pre_tanh_value,
        )

    @torch.jit.export
    def jit_forward(self, obs, latent_variable):
        h = self.mlp.jit_forward(obs, latent_variable)
        mean = self.last_fc(h)
        action = torch.tanh(mean)
        return action

    @torch.jit.ignore
    def get_log_prob_entropy(self, obs, latent_variable, acts):
        h = self.mlp(obs, latent_variable)
        mean = self.last_fc(h)

        if self.conditioned_std:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        else:
            log_std = self.action_log_std.expand_as(mean)

        tanh_normal = ReparamTanhMultivariateNormal(mean, log_std)
        log_prob = tanh_normal.log_prob(acts)

        entropy = (0.5 + 0.5 * math.log(2 * math.pi) + log_std).sum(
            dim=-1, keepdim=True
        )

        return log_prob, entropy

    @torch.jit.ignore
    def get_log_prob(self, obs, latent_variable, acts, return_normal_params=False):
        h = self.mlp(obs, latent_variable)
        mean = self.last_fc(h)

        if self.conditioned_std:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        else:
            log_std = self.action_log_std.expand_as(mean)

        tanh_normal = ReparamTanhMultivariateNormal(mean, log_std)
        log_prob = tanh_normal.log_prob(acts)

        if return_normal_params:
            return log_prob, mean, log_std
        return log_prob


class ReparamMultivariateGaussianPolicy(Mlp, ExplorationPolicy):
    def __init__(
        self,
        hidden_sizes,
        obs_dim,
        action_dim,
        conditioned_std=False,
        init_w=1e-3,
        **kwargs,
    ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs,
        )
        self.conditioned_std = conditioned_std

        if self.conditioned_std:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.action_log_std = nn.Parameter(torch.zeros(1, action_dim))

        self.last_fc.weight.data.mul_(0.1)
        self.last_fc.bias.data.mul_(0.0)

    def get_action(self, obs_np, deterministic=False):
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return self.eval_np(obs_np, deterministic=deterministic)[0]

    def forward(
        self, obs, deterministic=False, return_log_prob=False, return_normal=False
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)

        if self.conditioned_std:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        else:
            log_std = self.action_log_std.expand_as(mean)
        std = torch.exp(log_std)

        log_prob = None
        expected_log_prob = None
        mean_action_log_prob = None
        if deterministic:
            action = mean
        else:
            normal = ReparamMultivariateNormalDiag(mean, log_std)
            action = normal.sample()
            if return_log_prob:
                log_prob = normal.log_prob(action)

        # I'm doing it like this for now for backwards compatibility, sorry!
        if return_normal:
            return (
                action,
                mean,
                log_std,
                log_prob,
                expected_log_prob,
                std,
                mean_action_log_prob,
                normal,
            )
        return (
            action,
            mean,
            log_std,
            log_prob,
            expected_log_prob,
            std,
            mean_action_log_prob,
        )

    def get_log_prob_entropy(self, obs, acts):
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)

        if self.conditioned_std:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        else:
            log_std = self.action_log_std.expand_as(mean)

        normal = ReparamMultivariateNormalDiag(mean, log_std)
        log_prob = normal.log_prob(acts)

        entropy = (0.5 + 0.5 * math.log(2 * math.pi) + log_std).sum(
            dim=-1, keepdim=True
        )
        return log_prob, entropy

    def get_log_prob(self, obs, acts, return_normal_params=False):
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)

        if self.conditioned_std:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        else:
            log_std = self.action_log_std.expand_as(mean)

        normal = ReparamMultivariateNormalDiag(mean, log_std)
        log_prob = normal.log_prob(acts)

        if return_normal_params:
            return log_prob, mean, log_std
        return log_prob
