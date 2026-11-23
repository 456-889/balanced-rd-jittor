"""Optional experiment with CompressAI's full-width, pretrained M&S Hyperprior.

Run prepare_data.py first. Requires PyTorch, torchvision, CompressAI 1.2.4 and
NumPy. Uses the same source-photo split as the cross-framework experiment.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import compressai
import numpy as np
import torch
from compressai.zoo import mbt2018_mean
from torch.nn import functional as F

from balanced import Balancer
from torch_engine import gradient_gram


ROOT = Path(__file__).resolve().parent
LAMBDA = 0.0018
QUALITY = 1
SEED = 1701
LR = 5e-5
WEIGHT_FILE = "mbt2018-mean-1-e522738d.pth.tar"


def training_losses(model, x):
    output = model(x)
    n, _, h, w = x.shape
    rate = sum(-part.log2().sum() for part in output["likelihoods"].values()) / (n * h * w)
    distortion = LAMBDA * 255 ** 2 * F.mse_loss(output["x_hat"], x)
    return rate, distortion


@torch.no_grad()
def test_bitstreams(model, images, names):
    model.eval()
    model.update(force=True)
    rows = []
    for image, name in zip(images, names):
        x = image.unsqueeze(0)
        h, w = x.shape[-2:]
        padded = F.pad(x, (0, (-w) % 64, 0, (-h) % 64))
        coded = model.compress(padded)
        decoded = model.decompress(coded["strings"], coded["shape"])["x_hat"][..., :h, :w]
        reference = model(padded)["x_hat"].clamp(0, 1)[..., :h, :w]
        max_difference = (decoded - reference).abs().max().item()
        # Entropy decoding and evaluation-forward quantization must agree.
        if max_difference > 2e-5 or not torch.isfinite(decoded).all():
            raise AssertionError(f"round trip mismatch for {name}: {max_difference}")
        bytes_used = sum(len(s) for stream in coded["strings"] for s in stream)
        # Paper's image quality convention: round the reconstruction to RGB8.
        mse = F.mse_loss((255 * decoded).round(), (255 * x).round()).item()
        rows.append({"name": str(name), "payload_bytes": bytes_used,
                     "actual_bpp": 8.0 * bytes_used / (h * w),
                     "psnr_db": 10.0 * math.log10(255 ** 2 / mse),
                     "forward_decode_max_abs": max_difference})
    return {"mean_actual_bpp": float(np.mean([r["actual_bpp"] for r in rows])),
            "mean_psnr_db": float(np.mean([r["psnr_db"] for r in rows])),
            "by_image": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=8, help="updates per method")
    parser.add_argument("--threads", type=int, default=2, help="PyTorch CPU threads")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "full_model.json")
    parser.add_argument("--bridge", type=Path, help="export exact training tensors for full Jittor replay")
    args = parser.parse_args()
    captured = {}
    if args.bridge:
        args.bridge.mkdir(parents=True, exist_ok=True)
    if args.steps < 1 or args.threads < 1:
        parser.error("steps and threads must be positive")

    torch.set_num_threads(args.threads)
    with np.load(ROOT / "data" / "patches.npz") as archive:
        train = torch.from_numpy(archive["train"]).float() / 255.0
        test = torch.from_numpy(archive["test"]).float() / 255.0
        names = archive["labels"].tolist()
    # The hyperprior downsamples by 64 overall. Resize only the tiny training
    # crops. For the 96x96 held-out images, zero-pad and crop at decode time.
    train = F.interpolate(train, size=(64, 64), mode="bilinear", align_corners=False)
    schedule = np.random.RandomState(SEED).randint(len(train), size=(args.steps, 2))
    report = {"model": "CompressAI mbt2018_mean", "quality": QUALITY, "lambda": LAMBDA,
              "scope": "CPU PyTorch full-backbone check, single rate, short fine-tuning",
              "torch": torch.__version__, "compressai": compressai.__version__,
              "threads": args.threads, "seed": SEED, "lr": LR,
              "batch_size": 2, "steps": args.steps,
              "bpp_definition": "lengths of y and z rANS payloads / original pixels, excludes file headers",
              "methods": {}}
    for method in ("vanilla", "s2"):
        model = mbt2018_mean(quality=QUALITY, metric="mse", pretrained=True).cpu()
        if method == "vanilla":
            weight_path = Path(torch.hub.get_dir()) / "checkpoints" / WEIGHT_FILE
            report.update({"pretrained_sha256": hashlib.sha256(weight_path.read_bytes()).hexdigest(),
                           "N": model.N, "M": model.M,
                           "parameters": sum(p.numel() for p in model.parameters()),
                           "before": test_bitstreams(model, test, names)})
        # The aux quantiles locate coding tails and have no R/D gradient. The
        # baseline keeps their pretrained values during this short adaptation.
        params = [p for n, p in model.named_parameters() if not n.endswith(".quantiles")]
        optimizer = torch.optim.Adam(params, lr=LR)
        balancer = Balancer(method)
        records = []
        model.train()
        start = time.perf_counter()
        for i, indices in enumerate(schedule):
            # Match quantization noise across both methods, at each step.
            torch.manual_seed(SEED + i)
            optimizer.zero_grad(set_to_none=True)
            r, d = training_losses(model, train[indices.tolist()])
            if args.bridge and method == "s2":
                # Replay the official noise draw exactly in its original memory
                # layout. Verify the inputs and all three forward outputs before
                # saving the perturbations for reuse by a different framework.
                batch = train[indices.tolist()]
                with torch.no_grad():
                    torch.manual_seed(SEED+i)
                    canonical = model(batch)
                    noise = {}
                    def capture(key):
                        def quantize(inputs, mode, means=None):
                            if mode != "noise":
                                raise RuntimeError("Expected training quantizer.")
                            sample = torch.empty_like(inputs).uniform_(-0.5, 0.5)
                            noise[key] = sample
                            return inputs + sample
                        return quantize
                    from unittest.mock import patch
                    torch.manual_seed(SEED+i)
                    with patch.object(model.entropy_bottleneck, "quantize", capture("z")), patch.object(
                            model.gaussian_conditional, "quantize", capture("y")):
                        repeat = model(batch)
                    assert torch.equal(repeat["x_hat"], canonical["x_hat"])
                    for k in ("y", "z"):
                        assert torch.equal(repeat["likelihoods"][k], canonical["likelihoods"][k])
                    torch.testing.assert_close(
                        sum(-p.log2().sum() for p in repeat["likelihoods"].values()) / (len(batch)*64**2),
                        r.detach(), rtol=0, atol=0)
                    captured[f"x{i}"] = batch.numpy()
                    captured[f"indices{i}"] = indices
                    captured[f"z{i}"] = noise["z"].reshape(model.N,len(batch),1,1).permute(1,0,2,3).numpy()
                    captured[f"y{i}"] = noise["y"].numpy()
            gram = gradient_gram(params, r, d) if method == "s2" else None
            a, b = balancer.weights(r.item(), d.item(), gram)
            (a * r + b * d).backward()
            optimizer.step()
            records.append({"step": i + 1, "rate": r.item(), "distortion": d.item(),
                            "grad_rate_coefficient": a, "grad_distortion_coefficient": b})
        elapsed = time.perf_counter() - start
        report["methods"][method] = {"training_seconds": elapsed, "train": records,
                                     "after": test_bitstreams(model, test, names)}
        after = report["methods"][method]["after"]
        print(method, after["mean_actual_bpp"], after["mean_psnr_db"], flush=True)
    if args.bridge:
        np.savez(args.bridge / "s2_batches.npz", **captured)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()

