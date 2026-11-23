"""One matched S2 fine-tuning step on the published M&S architecture and weights."""
import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
LR = 5e-5
LAMBDA = 0.0018


def fixture():
    with np.load(ROOT / "data" / "patches.npz") as f:
        # Double a training-only crop in each spatial direction to 64 x 64.
        x = np.repeat(np.repeat(f["train"][:1], 2, 2), 2, 3).astype("float32") / 255
    rng = np.random.RandomState(4242)
    z = rng.uniform(-0.5, 0.5, (1, 128, 1, 1)).astype("float32")
    y = rng.uniform(-0.5, 0.5, (1, 192, 4, 4)).astype("float32")
    return x, z, y


def torch_step():
    import torch
    from compressai.zoo import mbt2018_mean
    from balanced import Balancer
    from torch_engine import gradient_gram
    torch.set_num_threads(2)
    model = mbt2018_mean(quality=1, metric="mse", pretrained=True).train()
    x, z_noise, y_noise = (torch.from_numpy(a) for a in fixture())
    # Bypass only the library's independent random draws. Its quantizer uses
    # x+noise while training, and the hyperlatent input is [C,1,B*H*W].
    model.entropy_bottleneck.quantize = lambda t, mode, means=None: t + z_noise.transpose(0, 1).reshape_as(t)
    model.gaussian_conditional.quantize = lambda t, mode, means=None: t + y_noise
    def objectives():
        out = model(x)
        r = sum(-p.log2().sum() for p in out["likelihoods"].values()) / (x.shape[2] * x.shape[3])
        d = (out["x_hat"] - x).square().mean() * (LAMBDA * 255**2)
        return r, d
    params = [p for n, p in model.named_parameters() if not n.endswith(".quantiles")]
    optimizer = torch.optim.Adam(params, lr=LR)
    r, d = objectives()
    gram = gradient_gram(params, r, d)
    a,b = Balancer("s2").weights(r.item(), d.item(), gram)
    (a*r + b*d).backward()
    optimizer.step()
    with torch.no_grad():
        r_new,d_new = objectives()
    return dict(rate=r.item(), distortion=d.item(), gram=gram.tolist(), a=a, b=b,
                post_rate=r_new.item(), post_distortion=d_new.item(),
                rd_parameter_count=sum(p.numel() for p in params))


def jittor_step(bridge):
    import jittor as jt
    from full_jittor_model import MeanScaleHyperprior
    from full_jittor_trainer import FullTrainer
    jt.flags.use_cuda = 0
    with np.load(bridge / "parameters.npz") as archive:
        model = MeanScaleHyperprior(archive, train=True)
    x, zn, yn = (jt.array(t).stop_grad() for t in fixture())
    trainer = FullTrainer(model, lr=LR, lam=LAMBDA)
    result = trainer.step(x, (zn, yn))
    with jt.no_grad():
        r, d = trainer.objectives(x, (zn, yn))
    result.update(post_rate=r.item(), post_distortion=d.item(),
                  rd_parameter_count=sum(p.numel() for p in trainer.params))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=("torch", "jittor"))
    parser.add_argument("--bridge", type=Path, required=True)
    args=parser.parse_args()
    if args.backend == "torch":
        result = torch_step()
    else:
        result = jittor_step(args.bridge)
        reference = json.loads((ROOT / "results" / "full_step_torch.json").read_text())
        for key in ("a","b","rate","distortion","post_rate","post_distortion","gram"):
            if not np.allclose(result[key], reference[key], rtol=2e-3, atol=3e-5):
                result.setdefault("failed_metrics", []).append(key)
        result["pass"] = not result.get("failed_metrics") and result["rd_parameter_count"] == reference["rd_parameter_count"]
    path = ROOT / "results" / f"full_step_{args.backend}.json"
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",encoding="utf-8")
    print(result, flush=True)
    if result.get("pass") is False:
        raise SystemExit("One-step mismatch; see saved diagnostics.")


if __name__ == "__main__":
    main()

