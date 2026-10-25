"""Strict local-file loaders for official POPE and SugarCrepe annotations.

Sources:
https://github.com/RUCAIBox/POPE (JSONL image/text/label/question_id)
https://github.com/RAIVNLab/sugar-crepe (JSON filename/caption/negative_caption)

These functions never download, synthesize ground truth, or infer human labels.
The caller must supply annotation files and an explicit image directory.
"""

import hashlib
import json
import random
from collections import Counter
from pathlib import Path


def _source(path, benchmark):
    path = Path(path)
    return {"benchmark": benchmark, "annotation_file": str(path.resolve()),
            "annotation_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "ground_truth": "supplied_annotation_file_not_model_generated"}


def _image_path(root, filename):
    root = Path(root).resolve()
    candidate = (root / filename).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("Image path escapes image_root") from exc
    return candidate


def load_pope(path, image_root, require_images=True, require_balanced=True):
    """Return (normalized_rows, provenance); retain whole source questions.

    IDs identify questions, while image_id permits image-clustered analyses.
    Missing image files are an error by default. Set require_images=False only
    when preparing/downloading a dataset; provenance records the missing count.
    """
    path = Path(path)
    source_rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not source_rows:
        raise ValueError("Empty POPE annotation file")
    rows, ids, missing = [], set(), set()
    for item in source_rows:
        if not all(key in item for key in ("question_id", "image", "text", "label")):
            raise ValueError("POPE requires question_id, image, text, label")
        question_id = str(item["question_id"])
        if question_id in ids:
            raise ValueError("Duplicate POPE question_id")
        ids.add(question_id)
        label = item["label"]
        if label not in ("yes", "no") or not isinstance(item["text"], str) or not item["text"].strip():
            raise ValueError("POPE labels must be explicit yes/no and questions nonempty")
        image = _image_path(image_root, item["image"])
        if not image.is_file():
            missing.add(str(image))
        rows.append({"id": question_id, "image_id": item["image"], "image": str(image),
                     "question": item["text"], "label": label})
    labels = Counter(r["label"] for r in rows)
    if not labels["yes"] or not labels["no"]:
        raise ValueError("POPE annotations require both yes and no labels")
    if require_balanced and labels["yes"] != labels["no"]:
        raise ValueError("POPE annotation labels are not balanced")
    if require_images and missing:
        raise FileNotFoundError("Missing %d POPE images; first: %s" % (len(missing), sorted(missing)[0]))
    return rows, {**_source(path, "POPE"), "n_questions": len(rows),
                  "n_images": len({r["image_id"] for r in rows}), "label_counts": dict(labels),
                  "missing_images": len(missing), "images_verified_present": not missing,
                  "balanced_labels": labels["yes"] == labels["no"]}


def load_sugarcrepe(path, image_root, category=None, require_images=True):
    """Return (rows, provenance) with labeled caption pairs, no answer ordering.

    Consumers must randomize/balance positive-caption presentation order for
    generative multiple-choice evaluation. CLIP evaluation compares positive
    and negative similarities directly and must count exact ties separately.
    """
    path = Path(path)
    source = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(source, dict) or not source:
        raise ValueError("SugarCrepe source must be a nonempty object keyed by item ID")
    category = category or path.stem
    rows, missing = [], set()
    for item_id, item in source.items():
        if not all(key in item for key in ("filename", "caption", "negative_caption")):
            raise ValueError("SugarCrepe requires filename, caption, negative_caption")
        positive, negative = item["caption"], item["negative_caption"]
        if not isinstance(positive, str) or not isinstance(negative, str) or not positive.strip() or not negative.strip() or positive.strip() == negative.strip():
            raise ValueError("SugarCrepe captions must be distinct nonempty strings")
        image = _image_path(image_root, item["filename"])
        if not image.is_file():
            missing.add(str(image))
        rows.append({"id": "%s:%s" % (category, item_id), "image_id": item["filename"],
                     "image": str(image), "positive": positive, "negative": negative,
                     "category": category})
    if require_images and missing:
        raise FileNotFoundError("Missing %d SugarCrepe images; first: %s" % (len(missing), sorted(missing)[0]))
    return rows, {**_source(path, "SugarCrepe"), "category": category, "n_pairs": len(rows),
                  "n_images": len({r["image_id"] for r in rows}), "missing_images": len(missing),
                  "images_verified_present": not missing}


def image_subset(rows, max_images, seed=42):
    """Deterministic image-level subset; retain every question/pair of each image."""
    if max_images < 1:
        raise ValueError("max_images must be positive")
    images = sorted({row["image_id"] for row in rows})
    random.Random(seed).shuffle(images)
    selected = set(images[:max_images])
    return [row for row in rows if row["image_id"] in selected]
