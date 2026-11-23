"""Jittor CPU codec and bias-corrected Adam used for the aligned comparison."""
import math

import jittor as jt
from jittor import nn
import numpy as np

jt.flags.use_cuda = 0


class Engine:
    def __init__(self, init, lr, lam):
        self.params = {name: jt.array(value).float32().start_grad() for name, value in init.items()}
        self.first = {name: jt.zeros_like(v).stop_grad() for name, v in self.params.items()}
        self.second = {name: jt.zeros_like(v).stop_grad() for name, v in self.params.items()}
        self.lr = lr
        self.lam = lam
        self.iteration = 0

    def _conv(self, x, name, stride=1):
        return nn.conv2d(x, self.params[name], self.params[name + "b"], stride=stride, padding=1)

    def forward(self, x, noise=None):
        x = jt.array(x).float32().stop_grad()
        y = self._conv(nn.relu(self._conv(x, "e1", 2)), "e2", 2)
        q = jt.round(y) if noise is None else y + jt.array(noise).float32().stop_grad()
        scale = jt.exp(self.params["log_scale"])
        prob = jt.sigmoid((q + 0.5) / scale) - jt.sigmoid((q - 0.5) / scale)
        n, _, h, w = x.shape
        rate = -jt.log(jt.maximum(prob, 1e-8)).float64().sum() / (n * h * w * math.log(2))
        d1 = nn.relu(self._conv(nn.interpolate(q, scale_factor=2, mode="nearest"), "d1"))
        x_hat = jt.sigmoid(self._conv(nn.interpolate(d1, scale_factor=2, mode="nearest"), "d2"))
        x_hat = x_hat[:, :, :h, :w]
        distortion = ((x_hat - x) ** 2).float64().mean() * (self.lam * 255.0 ** 2)
        return rate, distortion, x_hat

    def step(self, x, noise, balancer):
        rate, distortion, _ = self.forward(x, noise)
        r, d = rate.item(), distortion.item()
        params = list(self.params.values())
        if balancer.method == "s2":
            grad_r = jt.grad(rate, params, retain_graph=True)
            grad_d = jt.grad(distortion, params)
            # Reduce over every trainable tensor, including parameters that
            # belong only to the rate or distortion objective.
            gr = jt.concat([g.reshape((-1,)).float64() for g in grad_r])
            gd = jt.concat([g.reshape((-1,)).float64() for g in grad_d])
            rr, rd, dd = (gr * gr).sum(), (gr * gd).sum(), (gd * gd).sum()
            gram = jt.stack([rr, rd, rd, dd]).reshape((2, 2)).stop_grad()
            a, b = balancer.weights(r, d, gram)
            coeff = jt.array(np.asarray([a, b], dtype='float32')).stop_grad()
            grads = [coeff[0] * u + coeff[1] * v for u, v in zip(grad_r, grad_d)]
        else:
            a, b = balancer.weights(r, d)
            coeff = jt.array(np.asarray([a, b], dtype='float32')).stop_grad()
            grads = jt.grad(coeff[0] * rate + coeff[1] * distortion, params)
        self.iteration += 1
        t = self.iteration
        # Changing Python literals would make Jittor compile a new kernel per step.
        bias = jt.array(np.asarray([1 - 0.9 ** t, 1 - 0.999 ** t], dtype='float32')).stop_grad()
        # Match torch.optim.Adam (epsilon placed after bias correction).
        with jt.no_grad():
            for (name, param), grad in zip(self.params.items(), grads):
                m = self.first[name] * 0.9 + grad * 0.1
                v = self.second[name] * 0.999 + grad * grad * 0.001
                delta = (m / bias[0]) / (jt.sqrt(v / bias[1]) + 1e-8)
                self.first[name].assign(m)
                self.second[name].assign(v)
                param.assign(param - self.lr * delta).start_grad()
        # Moments feed the updated parameters. List them explicitly so Jittor
        # materializes the optimizer state instead of fusing earlier steps.
        states = list(self.first.values()) + list(self.second.values()) + list(self.params.values())
        jt.sync(states, True, False)
        if balancer.method == "s1":
            with jt.no_grad():
                post_r, post_d, _ = self.forward(x, noise)
            balancer.observe(post_r.item(), post_d.item())
        return r, d, float(a), float(b)

    def evaluate(self, x):
        with jt.no_grad():
            rate, distortion, xhat = self.forward(x)
            return rate.item(), distortion.item(), xhat.numpy().copy()

    def state(self):
        return {key: value.numpy().copy() for key, value in self.params.items()}
