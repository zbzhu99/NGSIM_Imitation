import torch
import torch.nn as nn
from rlkit.torch.core import PyTorchModule


class MlpPosterior(PyTorchModule):
    def __init__(
        self,
        obs_dim,
        action_dim,
        latent_distribution,
        num_layer_blocks=2,
        hid_dim=64,
        hid_act="relu",
        use_bn=False,
    ):
        self.save_init_params(locals())
        super().__init__()
        self.latent_distribution = latent_distribution
        self.dist_flat_dim = latent_distribution.dist_flat_dim

        if hid_act == "relu":
            hid_act_class = nn.ReLU
        elif hid_act == "tanh":
            hid_act_class = nn.Tanh
        else:
            raise NotImplementedError()

        self.mod_list = nn.ModuleList([nn.Linear(obs_dim + action_dim, hid_dim)])
        if use_bn:
            self.mod_list.append(nn.BatchNorm1d(hid_dim))
        self.mod_list.append(hid_act_class())

        for i in range(num_layer_blocks - 1):
            self.mod_list.append(nn.Linear(hid_dim, hid_dim))
            if use_bn:
                self.mod_list.append(nn.BatchNorm1d(hid_dim))
            self.mod_list.append(hid_act_class())

        self.mod_list.append(nn.Linear(hid_dim, self.dist_flat_dim))
        self.model = nn.Sequential(*self.mod_list)

    def get_logits(self, obs, action):
        inputs = torch.cat([obs, action], dim=-1)
        logits = self.model(inputs)
        return logits

    def get_log_posterior(self, obs, action, latent_variable):
        logits = self.get_logits(obs, action)
        dist_info = self.latent_distribution.activate_dist(logits)
        log_posterior = self.latent_distribution.logli(
            latent_variable, dist_info
        ).unsqueeze(-1)
        return log_posterior

    def get_posterior(self, obs, action, latent_variable):
        return torch.exp(self.get_log_posterior(obs, action, latent_variable))
