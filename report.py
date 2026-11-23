"""Build figures and alignment tables using all eight completed experiment runs."""
import csv
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
# Respect an explicit setting, or use the project drive's runtime when present.
local_runtime = ROOT.parent / "runtime"
if local_runtime.is_dir():
    os.environ.setdefault("MPLCONFIGDIR", str(local_runtime / "cache" / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

METHODS = ("vanilla", "s1", "vanilla-ft", "s2")
FRAMEWORKS = ("torch", "jittor")
LABELS = {"vanilla": "R+D", "s1": "S1", "vanilla-ft": "R+D fine-tune", "s2": "S2 fine-tune"}
COLORS = {"vanilla": "#376a99", "s1": "#bc5033", "vanilla-ft": "#487463", "s2": "#84589b"}
METRICS = ("bpp", "psnr_db", "rd", "median_train_step_ms", "inference_ms_per_image")


def load_runs(test_shape):
    runs = {}
    for method in METHODS:
        for fw in FRAMEWORKS:
            path = RESULTS / fw / method
            with (path / "metrics.json").open(encoding="utf-8") as f:
                metrics = json.load(f)
            train = np.atleast_1d(np.genfromtxt(path / "train.csv", delimiter=",", names=True))
            with np.load(path / "recon.npz") as f:
                recon = f["images"]
            with np.load(path / "checkpoint.npz") as f:
                params = dict(f)
            if recon.shape != test_shape or metrics["steps"] < 1:
                raise ValueError(f"{path}: unexpected image shape or empty run")
            expected = np.arange(1, metrics["steps"] + 1)
            np.testing.assert_array_equal(train["step"], expected)
            values = [recon, *params.values(), *(train[k] for k in train.dtype.names)]
            if not all(np.isfinite(v).all() for v in values) or not all(np.isfinite(metrics[k]) for k in METRICS):
                raise ValueError(f"{path}: nonfinite values")
            runs[method, fw] = (metrics, train, recon, params)
        tm = runs[method, "torch"][0]
        jm = runs[method, "jittor"][0]
        for field in ("schedule_sha256", "steps", "lambda", "seed", "lr", "batch_size", "warm_start"):
            if tm[field] != jm[field]:
                raise ValueError(f"{method}: settings differ for {field}")
    return runs


def store_vector(fig, name):
    for extension in ("png", "svg"):
        fig.savefig(FIGURES / f"{name}.{extension}", dpi=185, bbox_inches="tight")
    plt.close(fig)


def loss_curves(runs):
    # Framework lines often coincide; points and separate styles keep both visible.
    fig, axes = plt.subplots(2, 2, figsize=(9.7, 6.0), sharey="row", constrained_layout=True)
    for ax, method in zip(axes.flat, METHODS):
        for fw in FRAMEWORKS:
            log = runs[method, fw][1]
            is_ref = fw == "torch"
            ax.plot(log["step"], log["rd"], color="#586c82" if is_ref else COLORS[method],
                    linewidth=1.3 if is_ref else 1.05, linestyle="-" if is_ref else "--",
                    marker=None if is_ref else "o", markerfacecolor="white", markersize=3,
                    markevery=max(1, len(log)//9), label="PyTorch" if is_ref else "Jittor")
        ax.set(title=LABELS[method], xlabel="Training step", ylabel="R + weighted D")
        ax.spines[["right", "top"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
        ax.legend(loc="upper right", frameon=False, fontsize=8)
    fig.suptitle("Mini-batch rate–distortion loss: scratch (top) and fine-tuning (bottom)")
    store_vector(fig, "loss_curves")


def rd_points(runs):
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.1), constrained_layout=True)
    regular = Line2D([], [], color="#36485e", linestyle="none", marker="o",
                     markerfacecolor="none", markersize=8, label="PyTorch")
    overlap = Line2D([], [], color="#36485e", linestyle="none", marker="x",
                     markersize=6, label="Jittor")
    for ax, methods, title in zip(axes, (METHODS[:2], METHODS[2:]), ("From scratch", "Fine-tuning")):
        handles = [regular, overlap]
        for method in methods:
            color = COLORS[method]
            for fw in FRAMEWORKS:
                metrics = runs[method, fw][0]
                style = (dict(marker="o", facecolors="none", edgecolors=color, s=115)
                         if fw == "torch" else dict(marker="x", color=color, s=46))
                ax.scatter(metrics["bpp"], metrics["psnr_db"], **style)
            handles.append(Line2D([], [], color=color, lw=3, label=LABELS[method]))
        ax.set(title=title, xlabel="Estimated rate (bits/pixel)", ylabel="Pooled RGB PSNR (dB)")
        ax.margins(x=.33, y=.36)
        ax.ticklabel_format(style="plain", useOffset=False)
        ax.locator_params(nbins=5)
        ax.spines[["right", "top"]].set_visible(False)
        ax.grid(alpha=.15)
        ax.legend(handles=handles, fontsize=8, loc="upper center", ncol=2, frameon=False,
                  bbox_to_anchor=(.5, -.22))
    lam = runs["s1", "jittor"][0]["lambda"]
    fig.suptitle(f"Test operating points at one lambda ({lam:g})")
    store_vector(fig, "rd_points")


def reconstruction(runs, original, label):
    ref, jt = (runs["s1", fw][2][0].transpose(1, 2, 0) for fw in FRAMEWORKS)
    err_ref = np.abs(original - ref).mean(axis=-1)
    err_jt = np.abs(original - jt).mean(axis=-1)
    err_cross = np.abs(ref - jt).mean(axis=-1)
    error_scale = max(float(err_ref.max()), float(err_jt.max()), 1e-12)
    cross_scale = max(float(err_cross.max()), 1e-12)
    fig, axes = plt.subplots(2, 3, figsize=(10.3, 6.4), constrained_layout=True)
    for ax, arr, title in zip(axes[0], (original, ref, jt), ("Original", "PyTorch S1", "Jittor S1")):
        ax.imshow(np.clip(arr, 0., 1.), interpolation="nearest")
        ax.set_title(title)
    for ax, arr, title, limit in zip(axes[1], (err_ref, err_jt, err_cross),
            ("|Original − PyTorch|", "|Original − Jittor|", "|PyTorch − Jittor|"),
            (error_scale, error_scale, cross_scale)):
        img = ax.imshow(arr, cmap="magma", vmin=0, vmax=limit, interpolation="nearest")
        ax.set_title(title)
        bar = fig.colorbar(img, ax=ax, orientation="horizontal", shrink=.83, pad=.03)
        bar.set_label("Mean channel error (RGB 0–1)", fontsize=8)
        bar.ax.tick_params(labelsize=8)
    for ax in axes.flat:
        ax.set_axis_off()
    fig.suptitle(f"Same held-out image and coordinates: {label}  |  S1")
    fig.savefig(FIGURES / "reconstruction.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def alignment(runs):
    rows = []
    for method in METHODS:
        tm, tl, tr, tp = runs[method, "torch"]
        jm, jl, jr, jp = runs[method, "jittor"]
        if tp.keys() != jp.keys() or any(tp[k].shape != jp[k].shape for k in tp):
            raise ValueError(f"{method}: checkpoint tensors differ in name/shape")
        sample_diff = np.abs(tr.astype("float64") - jr)
        row = {"method": method, "steps": tm["steps"], "schedule_sha256": tm["schedule_sha256"]}
        for metric in METRICS:
            row["torch_" + metric] = tm[metric]
            row["jittor_" + metric] = jm[metric]
            row["jittor_minus_torch_" + metric] = jm[metric] - tm[metric]
        for field in ("rd", "rate_bpp", "weighted_distortion", "rate_coeff", "dist_coeff"):
            row["train_max_abs_" + field] = float(np.abs(tl[field] - jl[field]).max())
        row["test_recon_mean_abs"] = float(sample_diff.mean())
        row["test_recon_max_abs"] = float(sample_diff.max())
        row["checkpoint_max_abs"] = max(float(np.max(np.abs(tp[k] - jp[k]))) for k in tp)
        rows.append(row)
    with (FIGURES / "alignment.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# 跨框架实验结果", "",
             "方向统一为 Jittor 减 PyTorch。bpp 是概率模型估计码率，PSNR 由测试样本合并均方误差计算。", "",
             "| 方法 | 测试 Δbpp | 测试 ΔPSNR（dB） | 测试 Δ(R+D) | 重建像素平均绝对差（0–1） |",
             "|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['method']} | {r['jittor_minus_torch_bpp']:+.2e} | "
                     f"{r['jittor_minus_torch_psnr_db']:+.2e} | {r['jittor_minus_torch_rd']:+.2e} | "
                     f"{r['test_recon_mean_abs']:.2e} |")
    lines.extend(["", "## 同一训练阶段内的方法差值", "",
                  "下表在 Jittor 上比较同样步数、相同训练日程的两组方法。Δ 为平衡方法减常规训练，"
                  "R+D 越低越好。单个 λ 只能给出一个率失真工作点，无法据此计算 BD-rate。", "",
                  "| 对比 | Δbpp | ΔPSNR（dB） | Δ(R+D) | 训练步耗时比 |",
                  "|---|---:|---:|---:|---:|"])
    for baseline, balanced_name in (("vanilla", "s1"), ("vanilla-ft", "s2")):
        base, new = (runs[m, "jittor"][0] for m in (baseline, balanced_name))
        lines.append(f"| {LABELS[balanced_name]} − {LABELS[baseline]} | {new['bpp']-base['bpp']:+.5f} | "
                     f"{new['psnr_db']-base['psnr_db']:+.4f} | {new['rd']-base['rd']:+.5f} | "
                     f"{new['median_train_step_ms']/base['median_train_step_ms']:.2f}× |")
    lines.extend(["", "训练耗时取预热后每步中位数，推理耗时包含将重建结果返回主机。"
                  "不同运行环境的时间仅描述这次实验。完整训练差值、时延、权重差值见同目录 alignment.csv。"])
    (FIGURES / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    with np.load(ROOT / "data" / "patches.npz") as dataset:
        originals = dataset["test"]
        label = str(dataset["labels"][0])
    runs = load_runs(originals.shape)
    FIGURES.mkdir(parents=True, exist_ok=True)
    loss_curves(runs)
    rd_points(runs)
    reconstruction(runs, originals[0].transpose(1, 2, 0).astype("float32") / 255., label)
    alignment(runs)
    print(f"Rendered figures and cross-framework tables in {FIGURES}")


if __name__ == "__main__":
    main()
