import torch
import itertools
import numpy as np
from torch import distributions

TINY = 1e-8


class Distribution:
    @property
    def dist_flat_dim(self):
        """
        rtype: int
        """
        raise NotImplementedError

    @property
    def dim(self):
        """
        rtype: int
        """
        raise NotImplementedError

    @property
    def effective_dim(self):
        raise NotImplementedError

    def sample(self, dist_info):
        raise NotImplementedError

    def sample_prior(self, batch_size):
        return self.sample(self.prior_dist_info(batch_size))

    def prior_dist_info(self, batch_size):
        raise NotImplementedError


class Categorical(Distribution):
    def __init__(self, dim):
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    @property
    def dist_flat_dim(self):
        return self.dim

    @property
    def effective_dim(self):
        return 1

    def logli(self, x_var, dist_info):
        """
        Args:
            x_var: one_hot coding of latent variable
            dist_info: distribution
        Returns:
            (batch_size, )
        """
        prob = dist_info["prob"]
        assert prob.shape[-1] == self.dim
        assert len(prob.shape) == 2, f"prob.shape: {prob.shape}"
        return torch.sum(torch.log(prob) * x_var, dim=-1)

    def prior_dist_info(self, batch_size):
        prob = torch.ones((batch_size, self.dim), dtype=torch.float32) / self.dim
        return dict(prob=prob)

    def sample(self, dist_info):
        prob = dist_info["prob"]
        samples = distributions.OneHotCategorical(prob).sample()
        assert samples.shape == prob.shape
        return samples

    def activate_dist(self, flat_dit):
        prob = torch.softmax(flat_dit, dim=-1)
        return dict(prob=prob)

    def entropy(self, dist_info):
        prob = dist_info["prob"]
        assert len(prob.shape) == 2, f"prob.shape: {prob.shape}"
        entropy = torch.sum(prob * torch.log(prob + TINY), dim=-1)
        return entropy

    @property
    def dist_info_keys(self):
        return ["prob"]


class Gaussian(Distribution):
    def __init__(self, dim, fix_std=False):
        self._dim = dim
        self._fix_std = fix_std

    @property
    def dim(self):
        return self._dim

    @property
    def dist_flat_dim(self):
        # mean + std
        return self._dim * 2

    @property
    def effective_dim(self):
        return self._dim

    def logli(self, x_var, dist_info):
        mean = dist_info["mean"]
        std = dist_info["std"]
        normal = distributions.Normal(mean, std)
        log_prob = normal.log_prob(x_var).sum(dim=-1)
        # print(f"mean: {mean.shape}, std: {std.shape}, x_var: {x_var.shape}, log_prob: {log_prob.shape}")
        assert len(mean.shape) == 2, f"mean.shape: {mean.shape}"
        assert len(std.shape) == 2, f"std.shape: {std.shape}"
        return log_prob

    def prior_dist_info(self, batch_size):
        mean = torch.zeros((batch_size, self.dim))
        std = torch.ones((batch_size, self.dim))
        return dict(mean=mean, std=std)

    def sample(self, dist_info):
        mean = dist_info["mean"]
        std = dist_info["std"]
        normal = distributions.Normal(mean, std)
        samples = normal.rsample()
        return samples

    @property
    def dist_info_keys(self):
        return ["mean", "std"]

    def activate_dist(self, flat_dist):
        mean = flat_dist[:, : self.dim]
        if self._fix_std:
            std = torch.ones_like(mean)
        else:
            std = torch.sqrt(torch.exp(flat_dist[:, self.dim :]))
        return dict(mean=mean, std=std)


class Uniform(Gaussian):
    """
    This distribution will sample prior data from a uniform distribution,
    but the prior and posterior are still modeled as a Gaussian.
    """

    def sample_prior(self, batch_size):
        uniform = distributions.Uniform(low=-1, high=1)
        samples = uniform.sample((batch_size, self.dim))
        return samples


class Product(Distribution):
    def __init__(self, dists):
        """
        type dists: list[Distribution]
        """
        self._dists = dists

    @property
    def dists(self):
        return list(self._dists)

    @property
    def dim(self):
        return sum(x.dim for x in self.dists)

    @property
    def effective_dim(self):
        return sum(x.effective_dim for x in self.dists)

    @property
    def dims(self):
        return [x.dim for x in self.dists]

    @property
    def dist_flat_dims(self):
        return [x.dist_flat_dim for x in self.dists]

    @property
    def dist_flat_dim(self):
        return sum(x.dist_flat_dim for x in self.dists)

    @property
    def dist_info_keys(self):
        ret = []
        for idx, dist in enumerate(self.dists):
            for k in dist.dist_info_keys:
                ret.append("id_%d_%s" % (idx, k))
        return ret

    def split_dist_info(self, dist_info):
        ret = []
        for idx, dist in enumerate(self.dists):
            cur_dist_info = dict()
            for k in dist.dist_info_keys:
                cur_dist_info[k] = dist_info["id_%d_%s" % (idx, k)]
            ret.append(cur_dist_info)
        return ret

    def join_dist_infos(self, dist_infos):
        ret = dict()
        for idx, dist, dist_info_i in zip(itertools.count(), self.dists, dist_infos):
            for k in dist.dist_info_keys:
                ret["id_%d_%s" % (idx, k)] = dist_info_i[k]
        return ret

    def split_var(self, x):
        """
        Split the tensor variable or value info per component.
        """
        cum_dims = list(np.cumsum(self.dims))
        out = []
        for slice_from, slice_to, dist in zip([0] + cum_dims, cum_dims, self.dists):
            sliced = x[:, slice_from:slice_to]
            out.append(sliced)
        return out

    def join_vars(self, xs):
        """
        Join the per component tensor variables into a whole tensor
        """
        return torch.cat(xs, -1)

    def split_dist_flat(self, dist_flat):
        """
        Split flat dist info into per component
        """
        cum_dims = list(np.cumsum(self.dist_flat_dims))
        out = []
        for slice_from, slice_to, dist in zip([0] + cum_dims, cum_dims, self.dists):
            sliced = dist_flat[:, slice_from:slice_to]
            out.append(sliced)
        return out

    def prior_dist_info(self, batch_size):
        ret = []
        for dist_i in self.dists:
            ret.append(dist_i.prior_dist_info(batch_size))
        return self.join_dist_infos(ret)

    def activate_dist(self, dist_flat):
        ret = dict()
        for idx, dist_flat_i, dist_i in zip(
            itertools.count(), self.split_dist_flat(dist_flat), self.dists
        ):
            dist_info_i = dist_i.activate_dist(dist_flat_i)
            for k, v in dist_info_i.items():
                ret["id_%d_%s" % (idx, k)] = v
        return ret

    def sample(self, dist_info):
        ret = []
        for dist_info_i, dist_i in zip(self.split_dist_info(dist_info), self.dists):
            ret.append(dist_i.sample(dist_info_i))
        return torch.cat(ret, dim=-1)

    def sample_prior(self, batch_size):
        ret = []
        for dist_i in self.dists:
            ret.append(dist_i.sample_prior(batch_size))
        return torch.cat(ret, dim=-1)

    def logli(self, x_var, dist_info):
        ret = 0.0
        for x_i, dist_info_i, dist_i in zip(
            self.split_var(x_var), self.split_dist_info(dist_info), self.dists
        ):
            ret += dist_i.logli(x_i, dist_info_i)
        return ret
