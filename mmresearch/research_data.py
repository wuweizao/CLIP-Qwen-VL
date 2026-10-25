"""Prepare an isolated, versioned confirmation dataset without changing v1 files.

The original 64 evaluated test images are an exploration set. Confirmation rows
are drawn independently from the pinned public test split after excluding those
images by RGB-content SHA256. dHash matches are review flags, never evidence
that near-duplicate contamination has been eliminated.
"""
import hashlib
import os
from pathlib import Path

from .common import ROOT, digest, read_json, read_rows, save_json, save_rows

DATASET = "jxie/flickr8k"
REVISION = "56f58c967835f7c508d684f36bd7897cca9d7634"
INSTRUCTION = "Describe this image in one concise English sentence."


def image_id(image):
    image = image.convert("RGB")
    return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def dhash64(image):
    """Horizontal difference hash; robust similarity heuristic, not an ID."""
    from PIL import Image
    image = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(image.getdata())
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | int(pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return value


def near_duplicate_audit(split_rows, hashes, threshold=4, max_saved_pairs=2000):
    """Flag dHash-close cross-split pairs; do not automatically drop images."""
    if not 0 <= threshold <= 64:
        raise ValueError("dHash threshold must be between 0 and 64")
    entries = [(split, row["id"]) for split, rows in split_rows.items() for row in rows]
    flags = []
    count = 0
    for i, (left_split, left_id) in enumerate(entries):
        for right_split, right_id in entries[i + 1:]:
            if left_split == right_split:
                continue
            distance = (hashes[left_id] ^ hashes[right_id]).bit_count()
            if distance <= threshold:
                count += 1
                if len(flags) < max_saved_pairs:
                    flags.append({"left_split": left_split, "left_id": left_id,
                                  "right_split": right_split, "right_id": right_id,
                                  "hamming_distance": distance})
    return {"algorithm": "horizontal_dhash_64", "threshold": threshold,
            "flagged_pairs": count, "saved_pairs": len(flags),
            "pairs_truncated": count > len(flags), "pairs": flags,
            "interpretation": "Heuristic review flags only. No human verification, automatic removal, or claim of decontamination."}


def assert_disjoint(split_rows):
    seen = {}
    for split, rows in split_rows.items():
        ids = [r["id"] for r in rows]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate image inside split " + split)
        for identifier in ids:
            if identifier in seen:
                raise ValueError("Exact image overlap: %s / %s" % (seen[identifier], split))
            seen[identifier] = split


def instructions_for(rows, captions_per_image=2):
    if captions_per_image < 1:
        raise ValueError("captions_per_image must be positive")
    return [{"id": row["id"] + "-%s" % j, "image_id": row["id"], "image": row["image"],
             "instruction": INSTRUCTION, "answer": caption,
             "answer_source": "public_human_caption", "source_split": row["split"]}
            for row in rows for j, caption in enumerate(row["captions"][:captions_per_image])]


def _directory(cfg):
    path = Path(cfg.get("data_dir", "data/research_v2"))
    path = (ROOT / path).resolve()
    data_root = (ROOT / "data").resolve()
    if path == data_root or data_root not in path.parents:
        raise ValueError("Research data_dir must be a dedicated child of project data/")
    return path


def _stream_split(cfg, split, info):
    from datasets import load_dataset
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    files = [endpoint + "/datasets/" + cfg.get("dataset", DATASET) + "/resolve/" + info.sha + "/" + f.rfilename
             for f in info.siblings if f.rfilename.startswith("data/" + split + "-") and f.rfilename.endswith(".parquet")]
    if not files:
        raise ValueError("No parquet files found for official split " + split)
    rows = load_dataset("parquet", data_files={split: files}, split=split, streaming=True)
    return rows.shuffle(seed=int(cfg.get("seed", 42)), buffer_size=1024)


def prepare(cfg):
    """Build train/validation/confirmation/exploration manifests and audit.

    Defaults: 1024 official-train, 128 official-validation, 512 unused
    official-test images. Existing complete identical preparations are reused
    only after manifest hashes are checked. Different settings require another
    dedicated data_dir, preserving the first confirmation definition.
    """
    directory = _directory(cfg)
    sizes = {"train": int(cfg.get("train_images", 1024)),
             "validation": int(cfg.get("validation_images", 128)),
             "test": int(cfg.get("test_images", 512))}
    if any(n < 1 for n in sizes.values()):
        raise ValueError("All research split sizes must be positive")
    seed = int(cfg.get("seed", 42))
    captions_per_image = int(cfg.get("captions_per_train_image", 2))
    original_path = ROOT / "data/test.jsonl"
    if not original_path.exists():
        raise FileNotFoundError("Original exploration test manifest is required: " + str(original_path))
    original = read_rows(original_path)
    original_ids = sorted({r["id"] for r in original})
    if len(original) != len(original_ids):
        raise ValueError("Original exploration test manifest has duplicate IDs")
    snapshot = {"dataset": cfg.get("dataset", DATASET), "revision": cfg.get("dataset_revision", REVISION)}
    definition = {"snapshot": snapshot, "seed": seed, "shuffle_buffer": 1024,
                  "requested_counts": sizes, "captions_per_train_image": captions_per_image,
                  "original_test_manifest_sha256": digest(original_path)}
    existing_path = directory / "source.json"
    if existing_path.exists():
        existing = read_json(existing_path)
        if existing.get("definition") != definition:
            raise ValueError("Refusing to replace an established research split; choose a new data_dir")
        for filename, expected in existing["manifest_sha256"].items():
            if digest(directory / filename) != expected:
                raise ValueError("Research manifest changed: " + filename)
        print("Reusing verified research data:", directory, flush=True)
        return existing
    directory.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import HfApi
    from PIL import Image
    info = HfApi().dataset_info(snapshot["dataset"], revision=snapshot["revision"])
    if snapshot["revision"] and info.sha != snapshot["revision"]:
        raise ValueError("Resolved dataset revision differs from requested immutable revision")
    split_rows = {}
    hashes = {}
    # Pre-exclude explored test images even if an unexpected exact duplicate
    # appears in a public train/validation split.
    seen = set(original_ids)
    rejected = {"train": 0, "validation": 0, "test": 0}
    for split, count in sizes.items():
        selected = []
        for item in _stream_split(cfg, split, info):
            image = item["image"].convert("RGB")
            identifier = image_id(image)
            if identifier in seen:
                rejected[split] += 1
                continue
            captions = list(dict.fromkeys(str(item[k]).strip() for k in sorted(item)
                                         if k.startswith("caption") and item[k]))
            if not captions:
                raise ValueError("No human caption fields in dataset item")
            relative = "data/images/%s.png" % identifier
            image_path = ROOT / relative
            image_path.parent.mkdir(parents=True, exist_ok=True)
            if not image_path.exists():
                image.save(image_path)
            selected.append({"id": identifier, "image": relative, "captions": captions,
                             "split": split, "official_split": split,
                             "dataset_revision": info.sha})
            hashes[identifier] = dhash64(image)
            seen.add(identifier)
            if len(selected) % 128 == 0 or len(selected) == count:
                print("research %s: %d/%d" % (split, len(selected), count), flush=True)
            if len(selected) == count:
                break
        if len(selected) != count:
            raise ValueError("Only %d eligible %s images; requested %d" % (len(selected), split, count))
        split_rows[split] = selected
        save_rows(directory / (split + ".jsonl"), selected)
    exploration = [{**r, "split": "exploration", "official_split": "test",
                    "purpose": "previously_evaluated_exploration_only"} for r in original]
    split_rows["exploration"] = exploration
    for row in exploration:
        with Image.open(ROOT / row["image"]) as image:
            hashes[row["id"]] = dhash64(image)
    assert_disjoint(split_rows)
    save_rows(directory / "exploration.jsonl", exploration)
    instructions = instructions_for(split_rows["train"], captions_per_image)
    save_rows(directory / "instructions.jsonl", instructions)
    save_json(directory / "near_duplicate_audit.json", near_duplicate_audit(
        split_rows, hashes, int(cfg.get("near_duplicate_threshold", 4))))
    manifests = [s + ".jsonl" for s in split_rows] + ["instructions.jsonl", "near_duplicate_audit.json"]
    source = {"schema_version": 2, "definition": definition, "dataset": snapshot["dataset"],
              "revision": info.sha, "snapshot": snapshot, "seed": seed,
              "counts": {**{s: len(rows) for s, rows in split_rows.items()}, "instructions": len(instructions)},
              "original_test_exclusion": {"count": len(original_ids), "ids": original_ids,
                                           "reason": "Previously evaluated images cannot enter confirmation evaluation."},
              "selection_uses_predictions": False, "exact_duplicate_rejections": rejected,
              "near_duplicate_policy": "dHash <= threshold flags only; not automatically removed or human verified",
              "manifest_sha256": {name: digest(directory / name) for name in manifests}}
    save_json(existing_path, source)
    return source
