import hashlib
import random
import os
from .common import ROOT, save_json, save_rows


def prepare(cfg):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    info = HfApi().dataset_info(cfg["dataset"], revision=cfg.get("dataset_revision"))
    revision = info.sha
    seen = set()
    counts = {}
    for split, count in (("train", cfg["train_images"]), ("validation", cfg["validation_images"]), ("test", cfg["test_images"])):
        endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
        files = [endpoint + "/datasets/" + cfg["dataset"] + "/resolve/" + revision + "/" + f.rfilename
                 for f in info.siblings if f.rfilename.startswith("data/" + split + "-") and f.rfilename.endswith(".parquet")]
        dataset = load_dataset("parquet", data_files={split: files}, split=split, streaming=True)
        dataset = dataset.shuffle(seed=cfg["seed"], buffer_size=1024)
        rows = []
        for item in dataset:
            image = item["image"].convert("RGB")
            fingerprint = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
            if fingerprint in seen:
                continue
            captions = [str(item[k]).strip() for k in sorted(item) if k.startswith("caption") and item[k]]
            if not captions:
                raise ValueError("Dataset must expose caption fields; got %s" % list(item))
            seen.add(fingerprint)
            relative = "data/images/%s.png" % fingerprint
            path = ROOT / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path)
            rows.append({"id": fingerprint, "image": relative, "captions": list(dict.fromkeys(captions)), "split": split})
            if len(rows) == count:
                break
        if len(rows) != count:
            raise ValueError("Insufficient unique images in " + split)
        save_rows(ROOT / "data" / (split + ".jsonl"), rows)
        counts[split] = len(rows)
        if split == "train":
            instructions = [{"id": r["id"] + "-%s" % j, "image_id": r["id"], "image": r["image"],
                             "instruction": "Describe this image in one concise English sentence.", "answer": caption}
                            for r in rows for j, caption in enumerate(r["captions"][:cfg["captions_per_train_image"]])]
            save_rows(ROOT / "data/instructions.jsonl", instructions)
            counts["instructions"] = len(instructions)
    save_json(ROOT / "data/source.json", {"dataset": cfg["dataset"], "revision": revision, "seed": cfg["seed"], "counts": counts})


def training_subset(rows, fraction, seed):
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    rows = sorted(rows, key=lambda r: r["id"])
    random.Random(seed).shuffle(rows)
    return rows[:max(1, int(len(rows) * fraction))]
