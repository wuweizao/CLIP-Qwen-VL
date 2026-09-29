import hashlib
import json
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def save_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def config(path="configs/small.json"):
    cfg = read_json(path)
    for key in ("clip_model", "vl_model"):
        local = ROOT / "data/models" / cfg[key].split("/")[-1]
        if (local / "download_provenance.json").exists():
            cfg[key] = str(local)
    return cfg


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance(cfg):
    import importlib.metadata
    import platform
    import torch
    return {"config": cfg, "python": platform.python_version(),
            "dataset_source": read_json(ROOT / "data/source.json"),
            "model_sources": {key: read_json(Path(cfg[key]) / "download_provenance.json")
                              for key in ("clip_model", "vl_model")
                              if (Path(cfg[key]) / "download_provenance.json").exists()},
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "packages": {p: importlib.metadata.version(p) for p in
                         ("torch", "transformers", "peft", "datasets", "accelerate")},
            "manifests": {s: digest(ROOT / "data" / (s + ".jsonl"))
                          for s in ("train", "validation", "test")}}
