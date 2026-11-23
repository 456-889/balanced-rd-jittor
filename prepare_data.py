"""Prepare deterministic, photo-disjoint crops and common initial weights."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
TRAIN = ("camera.png", "coffee.png", "chelsea.png", "brick.png", "grass.png", "hubble_deep_field.jpg")
TEST = ("astronaut.png", "rocket.jpg", "gravel.png")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, help="scikit-image packaged data directory")
    args = parser.parse_args()
    if args.images is None:
        from skimage import data
        args.images = Path(data.__file__).parent
    out = ROOT / "data"
    out.mkdir(exist_ok=True)
    rng = np.random.RandomState(1701)
    train, test, labels = [], [], []
    manifest = {"train_photos": list(TRAIN), "test_photos": list(TEST), "sha256_original": {}}
    for name in TRAIN + TEST:
        path = args.images / name
        manifest["sha256_original"][name] = digest(path)
        with Image.open(path) as source:
            image = source.convert("RGB")
        if name in TRAIN:
            for _ in range(32):
                width = int(rng.choice([32, 48, 64, 96, 128]))
                x = rng.randint(image.width - width + 1)
                y = rng.randint(image.height - width + 1)
                tile = image.crop((x, y, x + width, y + width)).resize((32, 32), Image.Resampling.LANCZOS)
                if rng.randint(2):
                    tile = ImageOps.mirror(tile)
                train.append(np.asarray(tile))
        else:
            test.append(np.asarray(ImageOps.fit(image, (96, 96), method=Image.Resampling.LANCZOS)))
            labels.append(name.split(".")[0] + "-overview")
            for k, (fx, fy) in enumerate(((0.20, 0.20), (0.50, 0.48), (0.75, 0.70))):
                x = int(fx * (image.width - 96))
                y = int(fy * (image.height - 96))
                test.append(np.asarray(image.crop((x, y, x + 96, y + 96))))
                labels.append(name.split(".")[0] + "-crop" + str(k + 1))
    arrays = {key: np.stack(items).transpose(0, 3, 1, 2).copy() for key, items in (("train", train), ("test", test))}
    np.savez_compressed(out / "patches.npz", **arrays, labels=np.array(labels))
    weights = {}
    for name, shape in (("e1", (8, 3, 3, 3)), ("e2", (4, 8, 3, 3)), ("d1", (8, 4, 3, 3)), ("d2", (3, 8, 3, 3))):
        weights[name] = (rng.normal(size=shape) / np.sqrt(np.prod(shape[1:]))).astype("float32")
        weights[name + "b"] = np.full(shape[0], 0.02, dtype="float32")
    weights["log_scale"] = np.zeros((1, 4, 1, 1), dtype="float32")
    np.savez(out / "init.npz", **weights)
    manifest.update({"seed": 1701, "train_shape": list(arrays["train"].shape), "test_shape": list(arrays["test"].shape),
                     "test_labels": labels, "array_sha256": {k: hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()}})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(manifest["train_shape"], manifest["test_shape"])


if __name__ == "__main__":
    main()
