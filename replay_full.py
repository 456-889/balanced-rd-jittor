"""Replay the original training batches with Jittor and evaluate its trained state."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def run_jittor(bridge):
    import jittor as jt
    from full_jittor_model import MeanScaleHyperprior
    from full_jittor_trainer import FullTrainer

    jt.flags.use_cuda = 0
    with np.load(bridge / "parameters.npz") as arrays:
        model = MeanScaleHyperprior(arrays, train=True)
    trainer = FullTrainer(model)
    reference = json.loads((bridge / "torch-replay.json").read_text())
    records, differences = [], []
    start = time.perf_counter()
    with np.load(bridge / "s2_batches.npz") as batches:
        for i, torch_row in enumerate(reference["methods"]["s2"]["train"]):
            host = [batches[f"{k}{i}"] for k in ("x", "z", "y")]
            if not all(np.isfinite(value).all() for value in host):
                raise FloatingPointError(f"Nonfinite image/noise at training step {i+1}")
            data = [jt.array(value).stop_grad() for value in host]
            update = trainer.step(data[0], data[1:])
            record = {"step": i+1, "training_indices": batches[f"indices{i}"].tolist(), **update}
            records.append(record)
            difference = {
                "rate": abs(update["rate"]-torch_row["rate"]),
                "distortion": abs(update["distortion"]-torch_row["distortion"]),
                "a": abs(update["a"]-torch_row["grad_rate_coefficient"]),
                "b": abs(update["b"]-torch_row["grad_distortion_coefficient"]),
            }
            differences.append(difference)
            print("step", i+1, update["rate"], update["distortion"], "delta", difference, flush=True)
    np.savez(bridge / "jittor_finetuned.npz", **{k: v.numpy() for k, v in model.trainable.items()})
    report = {"steps": len(records), "unit": "full pretrained mean + scale hyperprior, PyTorch and Jittor, S2",
              "training_seconds_including_any_JIT": time.perf_counter()-start, "train": records,
              "sha256_batches": hashlib.sha256((bridge/"s2_batches.npz").read_bytes()).hexdigest(),
              "training_max_abs_difference": {k: max(row[k] for row in differences) for k in differences[0]}}
    images, predicted = [], []
    with np.load(ROOT / "data" / "patches.npz") as f, jt.no_grad():
        for raw, name in zip(f["test"], f["labels"]):
            original = raw[None].astype("float32") / 255
            h, w = original.shape[2:]
            x = np.pad(original, ((0,0),(0,0),(0,(-h)%64),(0,(-w)%64)))
            trace = model.forward(jt.array(x))
            rec = np.clip(trace["g_s.6"].numpy()[:, :, :h, :w], 0, 1)
            if not np.isfinite(rec).all():
                raise FloatingPointError(f"Nonfinite held-out reconstruction: {name}")
            predicted.append(rec[0])
            mse8 = np.mean((np.rint(255*rec)-raw[None])**2, dtype=np.float64)
            bits = -sum(np.log2(trace[k].numpy().astype("float64")).sum() for k in ("y_probability","z_probability"))
            if not np.isfinite([bits,mse8]).all() or mse8 <= 0:
                raise FloatingPointError(f"Invalid held-out rate or MSE: {name}")
            images.append({"name": str(name), "estimated_bpp": bits/(h*w),
                           "psnr_rgb8_db": 10*math.log10(255**2/mse8)})
    np.savez(bridge / "jittor_predictions.npz", reconstructed=np.stack(predicted))
    report["native_jittor_test_views"] = images
    (ROOT/"results"/"full_jittor_finetune.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")


def audit_bitstreams(bridge):
    import torch
    from compressai.zoo import mbt2018_mean
    from torch.nn import functional as F
    from validate_full_model import test_bitstreams
    torch.set_num_threads(2)
    model = mbt2018_mean(quality=1, metric="mse", pretrained=True).cpu().eval()
    with np.load(bridge/"jittor_finetuned.npz") as trained, torch.no_grad():
        params = {k: v for k,v in model.named_parameters() if not k.endswith(".quantiles")}
        assert set(params) == set(trained.files)
        for name, parameter in params.items():
            parameter.copy_(torch.from_numpy(trained[name]))
    with np.load(ROOT/"data"/"patches.npz") as data:
        test = torch.from_numpy(data["test"]).float()/255
        names = data["labels"].tolist()
    # Measure actual headers-excluded y/z bytes by coding the Jittor-trained
    # tensors using the independent compiled CompressAI rANS implementation.
    coded = test_bitstreams(model, test, names)
    native = json.loads((ROOT/"results"/"full_jittor_finetune.json").read_text())
    torch_reference = json.loads((bridge/"torch-replay.json").read_text())["methods"]["s2"]["after"]
    errors = []
    with np.load(bridge/"jittor_predictions.npz") as jp, torch.no_grad():
        for x, expected in zip(test, jp["reconstructed"]):
            h,w = x.shape[-2:]
            padded = F.pad(x[None],(0,(-w)%64,0,(-h)%64))
            enc = model.compress(padded)
            decoded = model.decompress(enc["strings"],enc["shape"])["x_hat"][0,:,:h,:w]
            errors.append(float(np.max(np.abs(decoded.numpy() - expected))))
    gap_bpp = abs(coded["mean_actual_bpp"]-torch_reference["mean_actual_bpp"])
    gap_db = abs(coded["mean_psnr_db"]-torch_reference["mean_psnr_db"])
    per_view = zip(coded["by_image"], torch_reference["by_image"])
    bytes_match = sum(a["payload_bytes"] == b["payload_bytes"] for a,b in per_view)
    report = {"views_with_equal_payload_lengths": bytes_match, "num_views": len(names), "coder": "CompressAI 1.2.4 rANS, using Jittor-fine-tuned parameters",
              "test": coded, "max_abs_decoded_vs_native_jittor": max(errors),
              "torch_finetune_mean_actual_bpp": torch_reference["mean_actual_bpp"],
              "torch_finetune_mean_psnr_db": torch_reference["mean_psnr_db"],
              "gap_in_mean_actual_bpp": gap_bpp, "gap_in_mean_psnr_db": gap_db,
              "training_max_abs_difference": native["training_max_abs_difference"]}
    # Local engineering tolerances for this short CPU experiment, separate
    # from the paper's BD-rate claims. Save actual errors even if a check fails.
    thresholds = {"training_rate": 2e-3, "training_distortion": 2e-3,
                  "training_weight": 2e-3, "decode_max_abs": 3e-4,
                  "mean_payload_bpp": 5e-4, "mean_psnr_db": 1e-2}
    differences = native["training_max_abs_difference"]
    report["absolute_tolerances"] = thresholds
    report["pass"] = bool(differences["rate"] <= thresholds["training_rate"]
                          and differences["distortion"] <= thresholds["training_distortion"]
                          and max(differences["a"], differences["b"]) <= thresholds["training_weight"]
                          and max(errors) <= thresholds["decode_max_abs"]
                          and gap_bpp <= thresholds["mean_payload_bpp"]
                          and gap_db <= thresholds["mean_psnr_db"])
    (ROOT/"results"/"full_jittor_bitstreams.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(report["test"]["mean_actual_bpp"],report["test"]["mean_psnr_db"],max(errors),gap_bpp,gap_db)
    if not report["pass"]:
        raise SystemExit("Full fine-tuning comparison exceeded a configured tolerance; see report.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("train", "bitstreams"),required=True)
    parser.add_argument("--bridge",type=Path,required=True)
    args=parser.parse_args()
    if args.stage=="train":
        run_jittor(args.bridge)
    else:
        audit_bitstreams(args.bridge)


if __name__=="__main__":
    main()

