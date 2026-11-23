"""Independent numerical checks of the CVPR 2025 balanced RD controllers.

python validate_balancer.py controller
python validate_balancer.py torch --upstream /path/to/official_train_balanced.py
python validate_balancer.py jittor
python validate_balancer.py compare

The last three commands compare an actual network training step with identical
data, weights, quantization noise and standard Adam on the network. Torch uses
the two original S1 loss/logit methods from the hash-pinned author source:
https://gitlab.com/viper-purdue/balanced-rd/-/raw/e6dacff161acbd42c5a705745bace48f1fec8dfb/train_balanced.py
Jittor uses its own codec autograd and network update plus our S1 controller.
The ELIC backbone and the release's cautious network update are outside this
one-step test. Run the two framework commands in their respective environments.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np

from balanced import Balancer

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
UPSTREAM_SHA = "e9e29e5036c070b5267aa361cfd4a5319416b03485014a03d29e6049e714f884"


def controller():
    """Check against native double-precision AD and a separate linear solver."""
    import torch

    rng = torch.Generator().manual_seed(4805)

    def sample():
        return torch.rand(2, dtype=torch.float64, generator=rng) * 2 + 0.001

    model = Balancer("s1")
    logits = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    adam = torch.optim.Adam([logits], lr=0.025)
    errors = np.zeros(3)

    for _ in range(90):
        # The release stacks [distortion, rate]; our API uses [rate, distortion].
        initial = sample().requires_grad_(True)
        after = initial.detach() * (sample() + 0.1)
        shifted = initial + 1 + 1e-8
        task_w = logits.softmax(0)
        objective = (shifted.log() * task_w / (task_w / shifted).sum().detach()).sum()
        coeff = torch.autograd.grad(objective, initial)[0].flip(0).numpy()
        got = model.weights(*initial.detach().flip(0).numpy())
        errors[0] = max(errors[0], np.max(np.abs(coeff - got)))
        delta = shifted.log().detach() - (after + 1 + 1e-8).log()
        grad_logits = torch.autograd.grad(task_w, logits, delta)[0]
        adam.zero_grad()
        logits.grad = grad_logits + 0.003 * logits.detach()
        adam.step()
        model.observe(*after.flip(0).numpy())
        errors[1] = max(errors[1], np.max(np.abs(model.logits - logits.detach().flip(0).numpy())))

        grad = torch.randn(2, 7, dtype=torch.float64, generator=rng)
        losses = sample()
        log_grad = grad / losses[:, None]
        q = log_grad @ log_grad.T
        u = torch.linalg.solve(q, torch.ones(2, dtype=torch.float64))
        raw = (u / u.sum()).softmax(0) / losses
        ref_coeff = (raw / raw.sum()).numpy()
        got = Balancer("s2").weights(*losses.numpy(), (grad @ grad.T).numpy())
        errors[2] = max(errors[2], np.max(np.abs(ref_coeff - got)))

    np.testing.assert_array_less(errors, 1e-12)
    for gram in (np.zeros((2, 2)), np.array([[1., 2.], [2., 4.]])):
        np.testing.assert_allclose(Balancer("s2").weights(1., 2., gram), [2/3, 1/3])
    np.testing.assert_allclose(Balancer("s2").weights(0, 0, np.zeros((2, 2))), [0.5, 0.5])
    assert Balancer("vanilla").weights(0, 0) == (1., 1.)
    names = ("S1 raw gradient coefficients", "S1 logit after Adam",
             "S2 raw gradient coefficients vs KKT solve")
    lines = ["float64 / CPU / 90 iterations / seed 4805", "Maximum absolute differences"]
    lines.extend(f"{name}: {error:.3e}" for name, error in zip(names, errors))
    lines.append("PASS (threshold 1e-12); equal/zero log-gradient and vanilla cases passed.")
    RESULTS.mkdir(exist_ok=True)
    text = "\n".join(lines) + "\n"
    (RESULTS / "controller_check.txt").write_text(text, encoding="utf-8")
    print(text, end="")


def official_weights(source):
    """Load only the two reviewed methods, without executing the upstream trainer."""
    import torch
    import torch.nn.functional as F

    body = source.read_bytes()
    if hashlib.sha256(body).hexdigest() != UPSTREAM_SHA:
        raise ValueError("Official source SHA-256 differs from the pinned revision")
    tree = ast.parse(body.decode("utf-8"), filename=str(source))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Balanced")
    wanted = ("get_weighted_loss", "update_task_weights")
    selected = [f for f in cls.body if isinstance(f, ast.FunctionDef) and f.name in wanted]
    module = ast.Module(body=selected, type_ignores=[])
    namespace = {"torch": torch, "F": F}
    exec(compile(module, str(source), "exec"), namespace)
    logits = torch.zeros(2, requires_grad=True)
    state = SimpleNamespace(w=logits, w_opt=torch.optim.Adam([logits], lr=0.025),
                            min_losses=torch.tensor([-1., -1.]), prev_loss=None, gamma=0.003)
    for name in wanted:
        setattr(state, name, MethodType(namespace[name], state))
    return state


def one_step(framework, source):
    rng = np.random.RandomState(4701)
    with np.load(ROOT / "data" / "init.npz") as file:
        init = dict(file)
    with np.load(ROOT / "data" / "patches.npz") as file:
        x = file["train"][:4].astype(np.float32) / 255.
    n, _, h, w = x.shape
    noise = rng.uniform(-.5, .5, size=(n, init["e2"].shape[0], (h+3)//4, (w+3)//4)).astype("float32")
    if framework == "torch":
        import torch
        from torch_engine import Engine

        engine = Engine(init, lr=0.001, lam=0.013)
        release = official_weights(source)
        rate, distortion, _ = engine.forward(x, noise)
        tasks = torch.stack([distortion, rate])
        loss = release.get_weighted_loss(tasks)
        coeff = torch.autograd.grad(loss, tasks, retain_graph=True)[0].detach().flip(0).numpy()
        before = [rate.item(), distortion.item()]
        engine.optimizer.zero_grad()
        loss.backward()
        engine.optimizer.step()
        with torch.no_grad():
            post_r, post_d, _ = engine.forward(x, noise)
            release.update_task_weights(torch.stack([post_d, post_r]))
        logits = release.w.detach().flip(0).numpy()
    else:
        from jittor_engine import Engine
        from jittor_balancer import Balancer as JittorBalancer

        engine = Engine(init, lr=0.001, lam=0.013)
        balancer = JittorBalancer("s1")
        rate, distortion, a, b = engine.step(x, noise, balancer)
        before, coeff = [rate, distortion], [a, b]
        logits = balancer.logits
        post_r, post_d, _ = engine.forward(x, noise)

    _, _, recon = engine.evaluate(x)
    values = {"losses_before": before, "coefficients": coeff, "logits_after": logits,
              "losses_after": [post_r.item(), post_d.item()], "reconstruction": recon}
    values.update({"param_" + key: v for key, v in engine.state().items()})
    output = RESULTS / "one_step"
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / (framework + ".npz"), **values)
    print(f"Saved {framework} S1 single-step outputs ({len(init)} parameter tensors)")


def compare():
    directory = RESULTS / "one_step"
    with np.load(directory / "torch.npz") as torch_run, np.load(directory / "jittor.npz") as jt_run:
        if set(torch_run.files) != set(jt_run.files):
            raise ValueError("One-step output keys do not match")
        result = {}
        for key in torch_run.files:
            ref, actual = torch_run[key], jt_run[key]
            np.testing.assert_equal(actual.shape, ref.shape)
            result[key] = float(np.max(np.abs(actual - ref)))
            np.testing.assert_allclose(actual, ref, rtol=2e-4, atol=2e-5, err_msg=key)
    summary = {"torch_reference": "two original upstream S1 methods + common network Adam",
               "official_source_sha256": UPSTREAM_SHA, "rtol": 2e-4, "atol": 2e-5,
               "seed_noise": 4701, "lambda": 0.013, "model_lr": 0.001,
               "passed": True, "max_abs_difference": result}
    (directory / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("controller", "torch", "jittor", "compare"))
    parser.add_argument("--upstream", type=Path, help="official train_balanced.py at the documented commit")
    args = parser.parse_args()
    if args.action == "controller":
        controller()
    elif args.action == "compare":
        compare()
    elif args.action == "torch" and args.upstream is None:
        parser.error("torch requires --upstream /path/to/official_train_balanced.py")
    else:
        one_step(args.action, args.upstream)
