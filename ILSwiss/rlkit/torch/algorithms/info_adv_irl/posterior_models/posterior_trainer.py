import torch.optim as optim
import torch
import rlkit.torch.utils.pytorch_util as ptu
from collections import OrderedDict
import numpy as np


class PosteriorTrainer:
    def __init__(
        self,
        posterior_model,
        post_lr=1e-3,
        post_momentum=0.9,
        soft_target_tau=1e-2,
        post_optimizer_class=optim.Adam,
    ):
        self.posterior_model = posterior_model
        self.post_lr = post_lr
        self.post_momentum = post_momentum
        self.soft_target_tau = soft_target_tau
        self.optimizer = post_optimizer_class(
            self.posterior_model.parameters(), lr=post_lr, betas=(post_momentum, 0.999)
        )
        self.target_posterior_model = posterior_model.copy()
        self.eval_statistics = None

    def train_step(self, batch):
        obs = batch["observations"]
        acts = batch["actions"]
        latents = batch["latents"]

        self.optimizer.zero_grad()
        self.posterior_model.train()
        log_posterior = self.posterior_model.get_log_posterior(obs, acts, latents)
        posterior_mean = torch.exp(log_posterior).mean()
        posterior_loss = -log_posterior.mean()
        posterior_loss.backward()
        self.optimizer.step()
        self.posterior_model.eval()

        self._update_target_network()

        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()
            self.eval_statistics[f"Posterior MLE Loss"] = np.mean(
                ptu.get_numpy(posterior_loss)
            )
            self.eval_statistics[f"Posterior Mean Likelihood"] = np.mean(
                ptu.get_numpy(posterior_mean)
            )

    def get_snapshot(self):
        return dict(
            posterior_model=self.posterior_model,
            target_posterior_model=self.target_posterior_model,
        )

    def _update_target_network(self):
        ptu.soft_update_from_to(
            self.posterior_model, self.target_posterior_model, self.soft_target_tau
        )

    def get_eval_statistics(self):
        return self.eval_statistics

    def end_epoch(self):
        self.eval_statistics = None

    @property
    def networks(self):
        return [self.posterior_model, self.target_posterior_model]
