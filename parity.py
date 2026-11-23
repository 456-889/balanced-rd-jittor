"""Cross-framework forward, objective coefficients and one Adam step at a common checkpoint."""
import argparse
import json
from pathlib import Path

import numpy as np

from train import ROOT, load_weights

OUT = ROOT / "results" / "parity"
METHODS = ("vanilla", "s1", "s2")


def measure(framework):
    if framework == "torch":
        from torch_engine import Engine
        from balanced import Balancer
    else:
        from jittor_engine import Engine
        from jittor_balancer import Balancer
    weights = load_weights(ROOT / "results" / "torch" / "vanilla" / "checkpoint.npz")
    with np.load(ROOT / "data" / "patches.npz") as data:
        x = data["train"][[0, 32, 64, 97]].astype("float32") / 255
    noise = np.random.RandomState(313).uniform(-0.5, 0.5, (4, 4, 8, 8)).astype("float32")
    values = {}
    for method in METHODS:
        engine = Engine(weights, 0.0006, 0.0035)
        balancer = Balancer(method)
        if method == "vanilla":
            rate, d, pixels = engine.evaluate(x)
            values["eval_rate_distortion"] = np.array([rate, d])
            values["eval_pixels"] = pixels
        values[method + "_rate_distortion_weights"] = np.array(engine.step(x, noise, balancer))
        values[method + "_parameters_after_update"] = np.concatenate([v.ravel() for v in engine.state().values()])
        if method == "s1":
            values["s1_post_update_logits"] = balancer.logits.copy()
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / (framework + ".npz"), **values)


def compare():
    with np.load(OUT / "torch.npz") as torch, np.load(OUT / "jittor.npz") as jt:
        delta = {k: float(np.max(np.abs(torch[k] - jt[k]))) for k in torch.files}
    thresholds = {k: (3e-4 if "rate_distortion" in k else 5e-5) for k in delta}
    failures = {k: d for k, d in delta.items() if not np.isfinite(d) or d > thresholds[k]}
    if failures:
        raise AssertionError(f"Common-checkpoint comparison failed: {failures}")
    result = {"max_absolute_difference": delta, "absolute_tolerances": thresholds, "passed": True,
              "common_checkpoint": "results/torch/vanilla/checkpoint.npz",
              "batch_indices": [0, 32, 64, 97], "noise_seed": 313}
    (OUT / "alignment.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("torch", "jittor", "compare"))
    action = parser.parse_args().action
    compare() if action == "compare" else measure(action)
