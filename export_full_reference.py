"""Export the official CompressAI M&S state and forward traces for Jittor."""
import argparse
from pathlib import Path

import numpy as np
import torch
from compressai.zoo import mbt2018_mean
from torch.nn import functional as F

from validate_full_model import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge", type=Path, required=True, help="directory for generated weights and traces")
    args = parser.parse_args()
    args.bridge.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    model = mbt2018_mean(quality=1, metric="mse", pretrained=True).cpu().eval()
    state = {k: v.cpu().numpy() for k, v in model.state_dict().items() if v.is_floating_point()}
    np.savez(args.bridge / "parameters.npz", **state)
    with np.load(ROOT / "data" / "patches.npz") as data:
        # One fixed overview per held-out source photograph, no training crops.
        selected = [i for i, s in enumerate(data["labels"]) if s.endswith("-overview")]
        images = torch.from_numpy(data["test"][selected]).float() / 255
        names = data["labels"][selected]
    with torch.no_grad():
        for i, (original, name) in enumerate(zip(images, names)):
            x = original.unsqueeze(0)
            h, w = x.shape[-2:]
            x = F.pad(x, (0, (-w) % 64, 0, (-h) % 64))
            trace = {"input": x, "original": original[None]}
            def capture(prefix, seq, tensor):
                for idx, layer in enumerate(seq):
                    tensor = layer(tensor)
                    # LeakyReLU operates in place in the official hyperprior.
                    trace[f"{prefix}.{idx}"] = tensor.clone()
                return tensor
            y = capture("g_a", model.g_a, x)
            z = capture("h_a", model.h_a, y)
            zh, pz = model.entropy_bottleneck(z)
            scales, means = capture("h_s", model.h_s, zh).chunk(2, dim=1)
            yh, py = model.gaussian_conditional(y, scales, means)
            xh = capture("g_s", model.g_s, yh)
            trace.update({"z_hat": zh, "y_hat": yh, "z_probability": pz, "y_probability": py})
            original_forward = model(x)
            assert torch.equal(original_forward["x_hat"], xh)
            assert torch.equal(original_forward["likelihoods"]["z"], pz)
            assert torch.equal(original_forward["likelihoods"]["y"], py)
            np.savez(args.bridge / f"trace-{i}.npz", **{k: v.numpy() for k, v in trace.items()}, name=name)
    print(f"Exported {len(state)} weight/buffer arrays and {len(names)} traces to {args.bridge}")


if __name__ == "__main__":
    main()

