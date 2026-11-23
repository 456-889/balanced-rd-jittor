"""Train/evaluate either framework using identical images, initialization and noise."""
import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
FIELDS = ("step", "rate_bpp", "weighted_distortion", "rd", "rate_coeff", "dist_coeff", "step_seconds")
METHODS = ("vanilla", "s1", "vanilla-ft", "s2")


def load_weights(path):
    with np.load(path) as bundle:
        return {key: bundle[key] for key in bundle.files}


def evaluate(engine, test, args, output):
    batch = test.astype("float32") / 255
    engine.evaluate(batch)  # Include synchronized host output, exclude first inference.
    times = []
    for _ in range(5):
        start = time.perf_counter()
        rate, distortion, recon = engine.evaluate(batch)
        times.append(time.perf_counter() - start)
    if not np.isfinite(recon).all() or not np.isfinite([rate, distortion]).all():
        raise FloatingPointError('non-finite test output')
    mse = np.mean((batch.astype("float64") - recon) ** 2, axis=(1, 2, 3))
    pooled_mse = float(mse.mean())
    psnr = -10 * math.log10(max(pooled_mse, 1e-12))
    np.savez_compressed(output / "recon.npz", images=recon)
    with (output / "test.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(("patch_index", "mse", "psnr_db"))
        for idx, item in enumerate(mse):
            writer.writerow((idx, item, -10 * math.log10(max(item, 1e-12))))
    return {"bpp": rate, "weighted_distortion": distortion, "rd": rate + distortion,
            "psnr_db": psnr, "mse": pooled_mse,
            "inference_ms_per_image": 1000 * float(np.median(times)) / len(batch),
            "test_images": len(test), "inference_repeats": len(times)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", required=True, choices=("jittor", "torch"))
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--test-only", action="store_true", help="load the trained checkpoint and evaluate")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    if args.framework == "torch":
        from torch_engine import Engine
        from balanced import Balancer
        import torch
        version = str(torch.__version__)
    else:
        from jittor_engine import Engine
        from jittor_balancer import Balancer
        import jittor
        version = str(jittor.__version__)
    finetune = args.method in ("vanilla-ft", "s2")
    steps = args.steps if args.steps is not None else (80 if finetune else 240)
    if steps < 1:
        parser.error("--steps must be positive")
    lr = 0.0006 if finetune else 0.003
    lam = 0.0035
    batch_size = 4
    seed = 1703 if finetune else 1702
    source = args.results / args.framework / "vanilla" / "checkpoint.npz" if finetune else ROOT / "data" / "init.npz"
    output = args.results / args.framework / args.method
    output.mkdir(parents=True, exist_ok=True)
    with np.load(ROOT / "data" / "patches.npz") as data:
        train, test = data["train"], data["test"]
    if args.test_only:
        engine = Engine(load_weights(output / "checkpoint.npz"), lr, lam)
        measured = evaluate(engine, test, args, output)
        print(json.dumps(measured, indent=2))
        return
    weights = load_weights(source)
    engine = Engine(weights, lr, lam)
    balancer = Balancer("vanilla" if args.method == "vanilla-ft" else args.method)
    rng = np.random.RandomState(seed)
    schedule_hash = hashlib.sha256()
    times = []
    with (output / "train.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(FIELDS)
        for step in range(1, steps + 1):
            idx = rng.randint(len(train), size=batch_size).astype("int32")
            x = train[idx].astype("float32") / 255
            noise = rng.uniform(-0.5, 0.5, (batch_size, 4, 8, 8)).astype("float32")
            schedule_hash.update(idx.tobytes())
            schedule_hash.update(noise.tobytes())
            start = time.perf_counter()
            rate, dist, a, b = engine.step(x, noise, balancer)
            seconds = time.perf_counter() - start
            if not np.isfinite([rate, dist, a, b]).all():
                raise FloatingPointError(f'non-finite training values at step {step}')
            times.append(seconds)
            writer.writerow((step, rate, dist, rate + dist, a, b, seconds))
            if step == 1 or step % 40 == 0 or step == steps:
                f.flush()
                print(f"{args.framework} {args.method} step={step} R={rate:.5f} D={dist:.5f} a={a:.5f}", flush=True)
    np.savez(output / "checkpoint.npz", **engine.state())
    metrics = evaluate(engine, test, args, output)
    warmup = min(10, steps - 1)
    metrics.update({"steps": steps, "batch_size": batch_size, "seed": seed, "lambda": lam, "lr": lr,
                    "framework": args.framework, "version": version, "method": args.method,
                    "warm_start": "vanilla/checkpoint.npz" if finetune else "data/init.npz",
                    "schedule_sha256": schedule_hash.hexdigest(), "timing_warmup_steps": warmup,
                    "median_train_step_ms": 1000 * float(np.median(times[warmup:])),
                    "python": sys.version.split()[0], "platform": platform.platform(), "device": "cpu"})
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
