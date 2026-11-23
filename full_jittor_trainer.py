"""Eight-step full-width native Jittor S2 reproduction using recorded PyTorch draws."""
import math

import numpy as np

import jittor as jt
from jittor_balancer import Balancer


class FullTrainer:
    def __init__(self, model, lr=5e-5, lam=0.0018):
        self.model = model
        self.params = list(model.trainable.values())
        self.first = [jt.zeros_like(p).stop_grad() for p in self.params]
        self.second = [jt.zeros_like(p).stop_grad() for p in self.params]
        self.iteration = 0
        self.lr = lr
        self.lam = lam
        self.balancer = Balancer("s2")

    def objectives(self, x, noise):
        out = self.model.forward(x, noise)
        n, _, h, w = x.shape
        rate = -(out["y_probability"].log().sum() + out["z_probability"].log().sum()) / (n*h*w*math.log(2))
        distortion = ((out["g_s.6"] - x)**2).mean() * (self.lam * 255**2)
        return rate, distortion

    def step(self, x, noise):
        rate, distortion = self.objectives(x, noise)
        r, d = rate.item(), distortion.item()
        if not np.isfinite([r, d]).all():
            raise FloatingPointError(f"Nonfinite training objective before step {self.iteration+1}: R={r}, D={d}")
        grad_r = jt.grad(rate, self.params, retain_graph=True)
        grad_d = jt.grad(distortion, self.params)
        names = list(self.model.trainable)
        for key, u, v in zip(names, grad_r, grad_d):
            if not np.isfinite(u.numpy()).all() or not np.isfinite(v.numpy()).all():
                raise FloatingPointError(f"Nonfinite raw gradient for {key} at step {self.iteration+1}")
        def dot(left, right):
            return sum((a.float64()*b.float64()).sum() for a,b in zip(left, right))
        rr, rd, dd = dot(grad_r,grad_r), dot(grad_r,grad_d), dot(grad_d,grad_d)
        gram = jt.stack([rr,rd,rd,dd]).reshape((2,2)).stop_grad()
        a,b = self.balancer.weights(r, d, gram)
        matrix = gram.numpy().tolist()
        if not np.isfinite(matrix).all() or not np.isfinite([a,b]).all():
            raise FloatingPointError("Nonfinite S2 Gram or gradient coefficient")
        self.iteration += 1
        t = self.iteration
        with jt.no_grad():
            grads = [(a*u+b*v).stop_grad() for u,v in zip(grad_r,grad_d)]
            # Materialize all gradients at the pre-update parameter values.
            jt.sync(grads, True, False)
            for p, g, m, v in zip(self.params, grads, self.first, self.second):
                m.assign(0.9*m + 0.1*g)
                v.assign(0.999*v + 0.001*g*g)
                # The CPU vectorized sqrt can produce NaN on positive
                # subnormal float32 values. sqrt(1e-34) is below half an ULP
                # at eps=1e-8, so this floor leaves the rounded Adam denominator
                # identical for every affected entry (including exact zero).
                v_hat = jt.maximum(v / (1-0.999**t), 1e-34)
                direction = (m / (1-0.9**t)) / (jt.sqrt(v_hat) + 1e-8)
                p.assign(p - self.lr*direction).start_grad()
            jt.sync(self.first + self.second + self.params, True, False)
            for key, p, m, v in zip(names, self.params, self.first, self.second):
                if not all(np.isfinite(var.numpy()).all() for var in (p,m,v)):
                    raise FloatingPointError(f"Nonfinite parameter or Adam moment at step {t}: {key}")
        return {"rate": r, "distortion": d, "gram": matrix, "a": a, "b": b}

