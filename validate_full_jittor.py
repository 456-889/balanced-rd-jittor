"""Verify all native Jittor M&S stages against the cached official PyTorch traces."""
import argparse
import json
import math
from pathlib import Path

import jittor as jt
import numpy as np

from full_jittor_model import MeanScaleHyperprior


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results" / "full_jittor.json")
    args = parser.parse_args()
    jt.flags.use_cuda = 0
    with np.load(args.bridge / "parameters.npz") as parameters, jt.no_grad():
        model = MeanScaleHyperprior(parameters)
    checks = []
    files = sorted(args.bridge.glob("trace-*.npz"))
    if not files:
        raise FileNotFoundError("Run export_full_reference.py first.")
    for path in files:
        with np.load(path) as ref, jt.no_grad():
            x = jt.array(ref["input"])
            expected = set(ref.files) - {"input", "original", "name"}
            trace = model.forward(x)
            assert expected == set(trace), expected ^ set(trace)
            layers = {}
            predicted = {}
            for k in trace:
                predicted[k] = trace[k].numpy()
                target = ref[k]
                assert predicted[k].shape == target.shape
                assert np.isfinite(predicted[k]).all()
                delta = np.abs(predicted[k].astype(np.float64) - target)
                layers[k] = {"max_abs": float(delta.max()), "mean_abs": float(delta.mean()),
                             "rel_l2": float(np.linalg.norm(delta.ravel()) /
                                             max(np.linalg.norm(target.ravel()), 1e-12))}
            def metrics(values):
                original = ref["original"]
                h, w = original.shape[-2:]
                rec = np.clip(values["g_s.6"][:, :, :h, :w], 0, 1)
                mse = np.mean((rec.astype(np.float64) - original) ** 2)
                bits = -sum(np.log2(values[k].astype(np.float64)).sum() for k in ("z_probability","y_probability"))
                return {"estimated_bpp": float(bits / (h*w)), "psnr_db_float": -10 * math.log10(mse)}
            actual, official = metrics(predicted), metrics(ref)
            record = {"image": str(ref["name"]), "torch": official, "jittor": actual, "layers": layers}
            record["pass"] = bool(max(v["rel_l2"] for v in layers.values()) < 2e-4
                                  and layers["g_s.6"]["max_abs"] < 3e-4
                                  and abs(actual["estimated_bpp"] - official["estimated_bpp"]) < 4e-5
                                  and abs(actual["psnr_db_float"] - official["psnr_db_float"]) < 3e-3)
            checks.append(record)
            print(record["image"], official, actual, "passed", record["pass"], flush=True)
    report = {"model": "mbt2018-mean quality=1, N=128 M=192", "jittor": jt.__version__,
              "torch_reference": "CompressAI 1.2.4 pretrained MSE", "checks": checks,
              "all_pass": all(row["pass"] for row in checks)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not report["all_pass"]:
        raise SystemExit("Parity check failed; see per-layer data in the report.")


if __name__ == "__main__":
    main()

