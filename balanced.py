"""Balanced rate--distortion gradient weights, with task order (rate, distortion).

Reference: Zhang, Duan, Huang and Zhu, CVPR 2025, pp. 2428--2438.
https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Balanced_Rate-Distortion_Optimization_in_Learned_Image_Compression_CVPR_2025_paper.html
Author implementation (PyTorch, S1):
https://gitlab.com/viper-purdue/balanced-rd/-/blob/e6dacff161acbd42c5a705745bace48f1fec8dfb/train_balanced.py

Pass R in bits/pixel and D with lambda already absorbed (e.g. lambda*255**2*MSE
for RGB floats in [0,1]). This controller returns detached scalar coefficients
for network autograd. The caller owns network parameters and their optimizer.

S1 follows the authors' released training code: each loss is shifted by
1+1e-8, and Adam updates the two logits. The paper's Algorithm 1 instead writes
an unshifted loss and plain gradient descent on the logits. The public ELIC
training script also uses clipping, a cautious Adam update on network weights,
and EMA. Those training-loop choices are independent of the weights below.
S2 follows paper Algorithm 2, Eqs. 20, 25, 26, 17--18. The cited repository
contains an S1 example; the S2 controller here is derived from the paper.
"""

import numpy as np


def _softmax(values):
    exp = np.exp(values - np.max(values))
    return exp / exp.sum()


def _loss_pair(rate, distortion):
    losses = np.asarray([rate, distortion], dtype=np.float64)
    if losses.shape != (2,) or not np.all(np.isfinite(losses)) or np.any(losses < 0):
        raise ValueError("rate and distortion must be finite, nonnegative scalars")
    return losses


class Balancer:
    """Compute gradient coefficients for vanilla R+D, S1 or S2."""

    def __init__(self, method, *, gamma=0.003, w_lr=0.025):
        if method not in ("vanilla", "s1", "s2"):
            raise ValueError("method must be 'vanilla', 's1' or 's2'")
        if not (np.isfinite(gamma) and gamma >= 0 and np.isfinite(w_lr) and w_lr > 0):
            raise ValueError("gamma must be nonnegative and w_lr must be positive")
        self.method = method
        self.gamma = gamma
        self.w_lr = w_lr
        self.logits = np.zeros(2, dtype=np.float64)
        self._first_moment = np.zeros(2, dtype=np.float64)
        self._second_moment = np.zeros(2, dtype=np.float64)
        self._steps = 0
        self._before = None

    def weights(self, rate, distortion, gram=None):
        """Return coefficients for a*grad(R) + b*grad(D), in that order.

        S2 needs the 2x2 Gram matrix of the two *raw* gradients over all
        trainable network parameters. A parameter absent from one objective
        contributes zero for that objective, including cross terms.
        """
        losses = _loss_pair(rate, distortion)
        if self.method == "vanilla":
            return 1.0, 1.0

        if self.method == "s1":
            losses = losses + 1.0 + 1e-8
            self._before = losses
            simplex_weights = _softmax(self.logits)
        else:
            # Protect the logarithm in the limiting case of exactly zero loss.
            losses = np.maximum(losses, 1e-12)
            raw_gram = np.asarray(gram, dtype=np.float64)
            if raw_gram.shape != (2, 2) or not np.all(np.isfinite(raw_gram)):
                raise ValueError("S2 needs a finite (2, 2) raw gradient Gram matrix")
            q = raw_gram / np.outer(losses, losses)
            cross = 0.5 * (q[0, 1] + q[1, 0])
            difference_norm_sq = q[0, 0] + q[1, 1] - 2.0 * cross
            # Eq. 25 with w_D=1-w_R. This scalar form also covers singular Q
            # with distinct log-gradients. With identical log-gradients every
            # mixture gives the same direction, so choose the symmetric one.
            scale = max(abs(q[0, 0]), abs(q[1, 1]), abs(cross))
            tolerance = 16.0 * np.finfo(np.float64).eps * scale
            rate_solution = 0.5
            if difference_norm_sq > tolerance:
                rate_solution = (q[1, 1] - cross) / difference_norm_sq
            simplex_weights = _softmax(np.array([rate_solution, 1.0 - rate_solution]))

        # Eqs. 16--18: express a normalized sum of log-gradients using raw ones.
        raw_weights = simplex_weights / losses
        raw_weights /= raw_weights.sum()
        return float(raw_weights[0]), float(raw_weights[1])

    def observe(self, rate_after, distortion_after):
        """After an S1 network step, re-evaluate both losses on the same batch."""
        if self.method != "s1":
            return
        if self._before is None:
            raise RuntimeError("call weights() before each S1 observe()")
        after = _loss_pair(rate_after, distortion_after) + 1.0 + 1e-8
        log_improvements = np.log(self._before) - np.log(after)
        w = _softmax(self.logits)
        # Jacobian(softmax)^T times the vector of log improvements (Eq. 15).
        gradient = w * (log_improvements - np.dot(w, log_improvements))
        gradient += self.gamma * self.logits

        self._steps += 1
        self._first_moment = 0.9 * self._first_moment + 0.1 * gradient
        self._second_moment = 0.999 * self._second_moment + 0.001 * gradient**2
        first = self._first_moment / (1.0 - 0.9**self._steps)
        second = self._second_moment / (1.0 - 0.999**self._steps)
        self.logits -= self.w_lr * first / (np.sqrt(second) + 1e-8)
        self._before = None
