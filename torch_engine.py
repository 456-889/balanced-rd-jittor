"""CPU PyTorch reference for the small learned image codec used in this experiment.

The analysis/synthesis transforms and factorized logistic prior keep the training
experiment small. The rate is the estimated entropy in bits per image pixel;
this codec does not write a compressed bitstream. Distortion is lambda * 255²
times MSE on RGB in [0, 1].
"""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def gradient_gram(params, rate, distortion):
    """Compute the two objective gradients' Gram matrix over network parameters."""
    params = tuple(params)
    grad_r = torch.autograd.grad(rate, params, retain_graph=True, allow_unused=True)
    grad_d = torch.autograd.grad(distortion, params, retain_graph=True, allow_unused=True)
    # A parameter used by just one objective has a zero cross term, but
    # its squared norm still contributes to the corresponding diagonal.
    rr = sum((g.double().square().sum() for g in grad_r if g is not None))
    dd = sum((g.double().square().sum() for g in grad_d if g is not None))
    rd = sum((a.double() * b.double()).sum() for a, b in zip(grad_r, grad_d)
             if a is not None and b is not None)
    return np.array([[float(rr), float(rd)], [float(rd), float(dd)]], dtype=np.float64)


class Engine:
    def __init__(self, init, lr, lam):
        # Tiny CPU convolutions are faster with one thread. This also makes
        # repeated timings less sensitive to the machine's thread count.
        torch.set_num_threads(1)
        self.params = {
            key: nn.Parameter(torch.tensor(value, dtype=torch.float32, device="cpu"))
            for key, value in init.items()
        }
        self.optimizer = torch.optim.Adam(self.params.values(), lr=lr, eps=1e-8)
        self.lam = lam

    def _conv(self, x, name, stride=1):
        return F.conv2d(x, self.params[name], self.params[name + "b"],
                        stride=stride, padding=1)

    def forward(self, x, noise=None):
        x = torch.as_tensor(x, dtype=torch.float32, device="cpu")
        y = self._conv(F.relu(self._conv(x, "e1", 2)), "e2", 2)
        if noise is None:
            q = torch.round(y)
        else:
            q = y + torch.as_tensor(noise, dtype=torch.float32, device="cpu")

        scale = self.params["log_scale"].exp()
        prob = torch.sigmoid((q + 0.5) / scale) - torch.sigmoid((q - 0.5) / scale)
        n, _, h, w = x.shape
        rate = -torch.log(prob.clamp_min(1e-8)).double().sum() / (n * h * w * math.log(2))

        d1 = F.relu(self._conv(F.interpolate(q, scale_factor=2, mode="nearest"), "d1"))
        x_hat = torch.sigmoid(self._conv(F.interpolate(d1, scale_factor=2, mode="nearest"), "d2"))
        # Stride-2 analysis uses ceiling division for odd input dimensions.
        x_hat = x_hat[:, :, :h, :w]
        distortion = (x_hat - x).square().double().mean() * (self.lam * 255.0 ** 2)
        return rate, distortion, x_hat

    def step(self, x, noise, balancer):
        self.optimizer.zero_grad(set_to_none=True)
        rate, distortion, _ = self.forward(x, noise)
        r, d = rate.detach().item(), distortion.detach().item()
        gram = gradient_gram(self.params.values(), rate, distortion) if balancer.method == "s2" else None
        a, b = balancer.weights(r, d, gram)
        (float(a) * rate + float(b) * distortion).backward()
        self.optimizer.step()
        if balancer.method == "s1":
            with torch.no_grad():
                post_r, post_d, _ = self.forward(x, noise)
            balancer.observe(post_r.item(), post_d.item())
        return r, d, float(a), float(b)

    @torch.no_grad()
    def evaluate(self, x):
        rate, distortion, x_hat = self.forward(x)
        return rate.item(), distortion.item(), x_hat.numpy().copy()

    def state(self):
        return {key: value.detach().numpy().copy() for key, value in self.params.items()}
