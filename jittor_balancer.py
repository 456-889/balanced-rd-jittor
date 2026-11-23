"""Native Jittor implementation of the two balanced R-D update rules."""
import numpy as np
import jittor as jt
from jittor import nn


class Balancer:
    def __init__(self, method, gamma=0.003, w_lr=0.025):
        if method not in ("vanilla", "s1", "s2"):
            raise ValueError("unknown balancing method")
        self.method = method
        self.gamma = gamma
        self.lr = w_lr
        self._logits = jt.zeros((2,), dtype="float64").start_grad()
        self.m = jt.zeros_like(self._logits).stop_grad()
        self.v = jt.zeros_like(self._logits).stop_grad()
        self.before = None
        self.t = 0

    @property
    def logits(self):
        return self._logits.numpy()

    @staticmethod
    def _losses(rate, distortion):
        return jt.array(np.array((rate, distortion), dtype="float64")).stop_grad()

    def weights(self, rate, distortion, gram=None):
        if self.method == "vanilla":
            return 1.0, 1.0
        losses = self._losses(rate, distortion)
        with jt.no_grad():
            if self.method == "s1":
                losses = losses + 1.0 + 1e-8
                self.before = losses
                w = nn.softmax(self._logits, dim=0)
            else:
                losses = jt.maximum(losses, 1e-12)
                q = gram / (losses.reshape((2, 1)) * losses.reshape((1, 2)))
                cross = 0.5 * (q[0, 1] + q[1, 0])
                norm = q[0, 0] + q[1, 1] - 2.0 * cross
                tol = 16.0 * np.finfo(np.float64).eps * jt.abs(q).max()
                # Degenerate identical gradients have a symmetric solution.
                valid = norm > tol
                denominator = jt.where(valid, norm, jt.ones_like(norm))
                first = jt.where(valid, (q[1, 1] - cross) / denominator, jt.ones_like(norm) * 0.5)
                w = nn.softmax(jt.concat([first.reshape((1,)), (1 - first).reshape((1,))]), dim=0)
            raw = w / losses
            values = (raw / raw.sum()).numpy()
        return float(values[0]), float(values[1])

    def observe(self, rate_after, distortion_after):
        if self.method != "s1":
            return
        after = self._losses(rate_after, distortion_after) + 1.0 + 1e-8
        improvement = jt.log(self.before) - jt.log(after)
        w = nn.softmax(self._logits, dim=0)
        objective = (w * improvement.stop_grad()).sum() + 0.5 * self.gamma * (self._logits ** 2).sum()
        grad = jt.grad(objective, self._logits)
        self.t += 1
        bias = jt.array(np.array([1 - 0.9 ** self.t, 1 - 0.999 ** self.t], dtype='float64')).stop_grad()
        with jt.no_grad():
            m = 0.9 * self.m + 0.1 * grad
            v = 0.999 * self.v + 0.001 * grad * grad
            update = (m / bias[0]) / (jt.sqrt(v / bias[1]) + 1e-8)
            self.m.assign(m)
            self.v.assign(v)
            self._logits.assign(self._logits - self.lr * update).start_grad()
        jt.sync([self.m, self.v, self._logits], True, False)
        self.before = None
