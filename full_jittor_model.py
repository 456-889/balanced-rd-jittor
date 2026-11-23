"""Native Jittor CPU forward pass of CompressAI 1.2.4 MeanScaleHyperprior.

The exported full-width parameters, learned entropy-bottleneck CDF, conditional
Gaussian densities, GDN and synthesis transforms match the reference network.
Coding payloads and their lengths are measured by validate_full_model.py.
"""
import math

import jittor as jt
from jittor import nn


class Erfc(jt.Function):
    """Complementary error function; 1-erf loses precision in Gaussian tails."""
    def execute(self, x):
        self.x = x
        flat = x.reshape((-1,))
        result = jt.code(flat.shape, flat.dtype, [flat], cpu_header="#include <cmath>",
                         cpu_src="for (int i=0; i<in0_shape0; ++i) { @out(i) = std::erfc(@in0(i)); }")
        return result.reshape(x.shape)

    def grad(self, grad):
        return grad * (-2 / math.sqrt(math.pi)) * jt.exp(-self.x * self.x)


class LowerBound(jt.Function):
    """CompressAI lower-bound backward lets a negative gradient re-enter range."""
    def execute(self, x, bound):
        self.keep = x >= bound
        return jt.maximum(x, bound)

    def grad(self, grad):
        return jt.where(self.keep | (grad < 0), grad, 0.0), None


def round_even(x):
    floor = jt.floor(x)
    fraction = x - floor
    odd = (jt.floor(floor / 2) * 2 != floor)
    return floor + ((fraction > 0.5) | ((fraction == 0.5) & odd)).cast(x.dtype)


class MeanScaleHyperprior:
    def __init__(self, state, train=False):
        self.p = {key: jt.array(value).stop_grad() for key, value in state.items()}
        # Aux quantiles and buffer constants do not receive RD gradients.
        self.trainable = {key: value for key, value in self.p.items()
                          if key.endswith((".weight", ".bias", ".gamma", ".beta"))
                          or key.startswith(("entropy_bottleneck._matrix", "entropy_bottleneck._bias",
                                             "entropy_bottleneck._factor"))}
        if train:
            for value in self.trainable.values():
                value.start_grad()

    def conv(self, x, key, stride=2, transpose=False):
        w, b = self.p[key + ".weight"], self.p[key + ".bias"]
        pad = w.shape[-1] // 2
        if transpose:
            return nn.conv_transpose2d(x, w, b, stride=stride, padding=pad, output_padding=stride-1)
        return nn.conv2d(x, w, b, stride=stride, padding=pad)

    def gdn(self, x, key, inverse):
        def nonnegative(name):
            bound = self.p[f"{key}.{name}_reparam.lower_bound.bound"]
            pedestal = self.p[f"{key}.{name}_reparam.pedestal"]
            return LowerBound()(self.p[f"{key}.{name}"], bound) ** 2 - pedestal
        channels = x.shape[1]
        gamma = nonnegative("gamma").reshape((channels, channels, 1, 1))
        norm = jt.sqrt(nn.conv2d(x * x, gamma, nonnegative("beta")))
        return x * norm if inverse else x / norm

    def transform(self, x, prefix, trace):
        for i in range(7 if prefix in ("g_a", "g_s") else 5):
            key = f"{prefix}.{i}"
            if i % 2:
                if prefix in ("g_a", "g_s"):
                    x = self.gdn(x, key, prefix == "g_s")
                else:
                    x = nn.leaky_relu(x, 0.01)
            else:
                unit_stride = (prefix == "h_a" and i == 0) or (prefix == "h_s" and i == 4)
                transpose = prefix == "g_s" or (prefix == "h_s" and i < 4)
                x = self.conv(x, key, stride=1 if unit_stride else 2, transpose=transpose)
            trace[key] = x
        return x

    def z_logits(self, x):
        for i in range(5):
            key = "entropy_bottleneck."
            m = nn.softplus(self.p[key + f"_matrix{i}"])
            x = nn.bmm(m, x) + self.p[key + f"_bias{i}"]
            if i != 4:
                x = x + jt.tanh(self.p[key + f"_factor{i}"]) * jt.tanh(x)
        return x

    def forward(self, x, noise=None):
        """Optional (z_noise, y_noise) provides matched uniform training samples."""
        trace = {}
        y = self.transform(x, "g_a", trace)
        z = self.transform(y, "h_a", trace)
        n, c, h, w = z.shape
        median = self.p["entropy_bottleneck.quantiles"][:, 0, 1].reshape((1, c, 1, 1))
        zh = round_even(z - median) + median if noise is None else z + noise[0]
        flat = zh.transpose((1, 0, 2, 3)).reshape((c, 1, -1))
        lo, hi = self.z_logits(flat - 0.5), self.z_logits(flat + 0.5)
        sign = ((lo + hi < 0).float32() - (lo + hi > 0).float32()).detach()
        pz = jt.abs(jt.sigmoid(sign * hi) - jt.sigmoid(sign * lo))
        pz = LowerBound()(pz, self.p["entropy_bottleneck.likelihood_lower_bound.bound"])
        pz = pz.reshape((c, n, h, w)).transpose((1, 0, 2, 3))
        params = self.transform(zh, "h_s", trace)
        channels = y.shape[1]
        scales = LowerBound()(params[:, :channels], self.p["gaussian_conditional.lower_bound_scale.bound"])
        means = params[:, channels:]
        yh = round_even(y - means) + means if noise is None else y + noise[1]
        absolute = jt.abs(yh - means)
        cdf_factor = -1 / math.sqrt(2)
        upper = 0.5 * Erfc()(cdf_factor * ((0.5 - absolute) / scales))
        lower = 0.5 * Erfc()(cdf_factor * ((-0.5 - absolute) / scales))
        py = LowerBound()(upper - lower, self.p["gaussian_conditional.likelihood_lower_bound.bound"])
        self.transform(yh, "g_s", trace)
        trace.update({"z_hat": zh, "y_hat": yh, "z_probability": pz, "y_probability": py})
        return trace

