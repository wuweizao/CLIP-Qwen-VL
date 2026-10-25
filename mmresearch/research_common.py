"""Paths, experiment identities and immutable manifests for research v2."""
import hashlib
import json
from pathlib import Path
from .common import ROOT, config, read_rows, read_json, save_json, digest


def load_config(path="configs/research.json"):
    return config(path)


def paths(cfg):
    return tuple(ROOT / cfg[key] for key in ("data_dir", "output_dir", "report_dir"))


def load_split(cfg, split):
    return read_rows(paths(cfg)[0] / (split + ".jsonl"))


def fingerprint(cfg):
    data, _, _ = paths(cfg)
    return {"config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
            "data_manifests": {split: digest(data / (split + ".jsonl")) for split in ("train", "validation", "test")}}


def run_name(regime, seed, sampling="random", count=None, scope="language", budget="epochs"):
    name = "%s_s%d_%s" % (regime, seed, sampling)
    if count is not None:
        name += "_n%d" % count
    return name + "_%s_%s" % (scope, budget)


def context_path(cfg, split, bank="full"):
    return paths(cfg)[1] / "contexts" / (split + "_" + bank + ".json")


def assert_completed(directory, identity):
    directory = Path(directory)
    path = directory / "completed.json"
    if not path.exists():
        return False
    if read_json(path)["identity"] != identity:
        raise ValueError("Existing completed artifact has different identity: " + str(directory))
    return True
