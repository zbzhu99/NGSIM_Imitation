import torch
import torch.nn as nn
import torch.nn.functional as F
from rlkit.torch.core import PyTorchModule


class MlpPosterior(PyTorchModule):
    def __init__(
        self,
        obs_dim,
        action_dim,
        latent_variable_num,
        num_layer_blocks=2,
        hid_dim=64,
        hid_act="relu",
        use_bn=False,
    ):
        self.save_init_params(locals())
        super().__init__()

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

        self.mod_list.append(nn.Linear(hid_dim, latent_variable_num))
        self.model = nn.Sequential(*self.mod_list)

    def get_posterior_dist(self, obs, action):
        input_ = torch.cat([obs, action], dim=-1)
        logits = self.model(input_)
        dist = F.softmax(logits, dim=-1)
        return dist

    def get_posterior(self, obs, action, latent_variable):
        dist = self.get_posterior_dist(obs, action)
        assert (
            len(latent_variable.shape) == 2 and len(dist.shape) == 2
        ), f"{latent_variable.shape}, {dist.shape}"
        posterior = torch.gather(dist, dim=-1, index=latent_variable.long())
        return posterior

    def get_log_posterior(self, obs, action, latent_variable):
        posterior = self.get_posterior(obs, action, latent_variable)
        log_posterior = torch.log(posterior)
        return log_posterior
