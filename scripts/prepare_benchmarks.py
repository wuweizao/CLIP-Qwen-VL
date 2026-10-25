"""Prepare pinned official SugarCrepe and original POPE diagnostic subsets.

Only original annotations and official COCO image URLs supply labels/images.
SugarCrepe-positive transfer captions are benchmark-selected references, not a
representative COCO evaluation set or independently verified human labels.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from io import BytesIO
import json
from pathlib import Path
import random
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mmresearch.common import ROOT, read_json, save_json, save_rows

SUGAR_REPO = "RAIVNLab/sugar-crepe"
SUGAR_COMMIT = "0047054b243992f0fad63d6f64f7544862daf846"
POPE_REPO = "AoiDragon/POPE"
POPE_COMMIT = "08d957b917e5a378a2f99d35b6293c536a66298b"
CATEGORIES = ("add_att", "add_obj", "replace_att", "replace_obj", "replace_rel", "swap_att", "swap_obj")
VARIANTS = ("random", "popular", "adversarial")


def stable_seed(seed, key):
    return int.from_bytes(hashlib.sha256((str(seed) + ":" + key).encode()).digest()[:8], "big")


def parse_pope(data):
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    seen = set()
    for row in rows:
        if not all(key in row for key in ("question_id", "image", "text", "label")):
            raise ValueError("Invalid original POPE annotation schema")
        if row["question_id"] in seen or row["label"] not in ("yes", "no"):
            raise ValueError("Duplicate POPE question ID or invalid label")
        if not re.fullmatch(r"COCO_val2014_\d{12}\.jpg", row["image"]):
            raise ValueError("Unexpected POPE official image filename")
        seen.add(row["question_id"])
    if not rows:
        raise ValueError("Empty POPE annotation")
    return rows


def select_sugar(source, category, count=16, seed=20261025):
    if category not in CATEGORIES or count < 1 or len(source) < count:
        raise ValueError("Invalid SugarCrepe category or subset size")
    ids = sorted(source)
    random.Random(stable_seed(seed, category)).shuffle(ids)
    selected = []
    for item_id in ids[:count]:
        item = source[item_id]
        if not re.fullmatch(r"\d{12}\.jpg", item.get("filename", "")):
            raise ValueError("Unexpected SugarCrepe COCO image filename")
        positive, negative = item.get("caption"), item.get("negative_caption")
        if not isinstance(positive, str) or not isinstance(negative, str) or not positive.strip() or not negative.strip() or positive.strip() == negative.strip():
            raise ValueError("Invalid SugarCrepe caption pair")
        selected.append({"id": "sugarcrepe:%s:%s" % (category, item_id), "image_id": item["filename"],
                         "image": "data/benchmark_images/val2017/" + item["filename"],
                         "positive": positive.strip(), "negative": negative.strip(),
                         "type": category, "category": category, "source_item_id": str(item_id),
                         "sourcecommit": SUGAR_COMMIT, "source_repo": SUGAR_REPO,
                         "label_source": "official_benchmark_annotation"})
    return selected


def select_pope(sources, questions=120, seed=20261025):
    """Use the same image set and all six questions/image in each variant."""
    if questions < 6 or questions % 6:
        raise ValueError("POPE question count must be a positive multiple of six")
    if set(sources) != set(VARIANTS):
        raise ValueError("All three original POPE variants are required")
    grouped = {}
    for variant in VARIANTS:
        groups = defaultdict(list)
        for row in sources[variant]:
            groups[row["image"]].append(row)
        grouped[variant] = groups
    common = set.intersection(*(set(grouped[v]) for v in VARIANTS))
    eligible = sorted(image for image in common if all(len(grouped[v][image]) == 6 and
                      Counter(r["label"] for r in grouped[v][image]) == {"yes": 3, "no": 3} for v in VARIANTS))
    random.Random(stable_seed(seed, "pope_images")).shuffle(eligible)
    image_count = questions // 6
    if len(eligible) < image_count:
        raise ValueError("Insufficient common balanced POPE image groups")
    images = eligible[:image_count]
    selected = []
    for variant in VARIANTS:
        for image in images:
            for row in sorted(grouped[variant][image], key=lambda r: r["question_id"]):
                selected.append({"id": "pope:%s:%s" % (variant, row["question_id"]), "image_id": image,
                                 "image": "data/benchmark_images/val2014/" + image,
                                 "question": row["text"], "label": row["label"], "variant": variant,
                                 "source_question_id": row["question_id"], "sourcecommit": POPE_COMMIT,
                                 "source_repo": POPE_REPO, "label_source": "official_benchmark_annotation"})
    return selected


def transfer_rows(sugar_rows):
    grouped = {}
    for row in sugar_rows:
        entry = grouped.setdefault(row["image_id"], {"id": "coco2017:" + row["image_id"],
            "image_id": row["image_id"], "image": row["image"], "captions": [], "split": "transfer",
            "reference_source": "SugarCrepe official positive captions; human authorship not independently verified",
            "selection_scope": "hard-negative-benchmark-conditioned subset; not representative COCO",
            "sourcecommit": SUGAR_COMMIT, "source_repo": SUGAR_REPO})
        if row["positive"] not in entry["captions"]:
            entry["captions"].append(row["positive"])
    return [grouped[key] for key in sorted(grouped)]


def _get(url, attempts=3):
    import requests
    for attempt in range(attempts):
        try:
            response = requests.get(url, timeout=(15, 45), headers={"User-Agent": "CLIP-Qwen-VL-research/2"})
            response.raise_for_status()
            return response.content
        except requests.RequestException:
            if attempt + 1 == attempts:
                raise
            time.sleep(1 + attempt)


def _tree(repo, revision):
    url = "https://api.github.com/repos/%s/git/trees/%s?recursive=1" % (repo, revision)
    data = json.loads(_get(url))
    if data.get("sha") != revision or data.get("truncated"):
        raise ValueError("Unexpected or truncated pinned GitHub source tree")
    return {r["path"]: r for r in data["tree"]}, url


def _annotation(repo, commit, filename, tree, output):
    url = "https://raw.githubusercontent.com/%s/%s/%s" % (repo, commit, filename)
    path = output / "annotations" / (repo.split("/")[-1] + "_" + Path(filename).name)
    if filename not in tree:
        raise ValueError("Pinned annotation missing: " + filename)
    data = path.read_bytes() if path.exists() else _get(url)
    blob_sha = hashlib.sha1(("blob %d\0" % len(data)).encode() + data).hexdigest()
    if blob_sha != tree[filename]["sha"]:
        raise ValueError("Pinned Git blob mismatch for " + filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(data)
    return data, {"repo": repo, "commit": commit, "path": filename, "url": url,
                  "git_blob_sha": blob_sha, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def _image(relative):
    from PIL import Image
    path = (ROOT / relative).resolve()
    allowed = (ROOT / "data/benchmark_images").resolve()
    if allowed not in path.parents:
        raise ValueError("Image download destination escapes benchmark_images")
    year, filename = path.parent.name, path.name
    urls = ["https://images.cocodataset.org/%s/%s" % (year, filename),
            "http://images.cocodataset.org/%s/%s" % (year, filename)]
    if path.exists():
        data = path.read_bytes()
        used = "cached_official_image"
    else:
        failures = []
        for url in urls:
            try:
                data = _get(url)
                with Image.open(BytesIO(data)) as image:
                    image.verify()
                used = url
                break
            except Exception as exc:
                failures.append(type(exc).__name__ + ": " + str(exc))
        else:
            raise RuntimeError("Official COCO download failed: %s; %s" % (relative, "; ".join(failures)))
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".part")
        temporary.write_bytes(data)
        temporary.replace(path)
    with Image.open(BytesIO(data)) as image:
        image.verify()
    return {"image": relative, "url": used, "official_urls": urls,
            "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def prepare(data_dir="data/research_v2", seed=20261025, sugar_per_type=16, pope_questions=120):
    output = (ROOT / data_dir).resolve()
    if (ROOT / "data").resolve() not in output.parents:
        raise ValueError("data_dir must be a dedicated child of project data/")
    output.mkdir(parents=True, exist_ok=True)
    definition = {"seed": seed, "sugar_per_type": sugar_per_type, "pope_questions_per_variant": pope_questions,
                  "sugar_commit": SUGAR_COMMIT, "pope_commit": POPE_COMMIT}
    status_path = output / "benchmark_source.json"
    if status_path.exists():
        prior = read_json(status_path)
        if prior.get("definition") != definition:
            raise ValueError("Refusing to replace an established benchmark subset with different settings")
    status = {"status": "preparing", "definition": definition, "sources": [], "download_failures": []}
    save_json(status_path, status)
    try:
        sugar_tree, sugar_api = _tree(SUGAR_REPO, SUGAR_COMMIT)
        pope_tree, pope_api = _tree(POPE_REPO, POPE_COMMIT)
        status["source_tree_urls"] = [sugar_api, pope_api]
        sugar = []
        for category in CATEGORIES:
            data, source = _annotation(SUGAR_REPO, SUGAR_COMMIT, "data/" + category + ".json", sugar_tree, output)
            status["sources"].append(source)
            sugar.extend(select_sugar(json.loads(data), category, sugar_per_type, seed))
        pope_sources = {}
        for variant in VARIANTS:
            data, source = _annotation(POPE_REPO, POPE_COMMIT,
                "output/coco/coco_pope_" + variant + ".json", pope_tree, output)
            status["sources"].append(source)
            pope_sources[variant] = parse_pope(data)
        pope = select_pope(pope_sources, pope_questions, seed)
        transfer = transfer_rows(sugar)
        image_paths = sorted({row["image"] for row in sugar + pope})
        status["requested_image_count"] = len(image_paths)
        save_json(status_path, status)
        records = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(_image, relative): relative for relative in image_paths}
            for future in as_completed(futures):
                relative = futures[future]
                try:
                    records.append(future.result())
                except Exception as exc:
                    status["download_failures"].append({"image": relative, "error": str(exc)})
                if len(records) % 16 == 0:
                    print("Benchmark images: %d/%d" % (len(records), len(image_paths)), flush=True)
        if status["download_failures"]:
            raise RuntimeError("%d official image downloads failed; no substituted images or evaluation manifests written" % len(status["download_failures"]))
        for filename, rows in (("sugarcrepe.jsonl", sugar), ("pope.jsonl", pope), ("transfer.jsonl", transfer)):
            save_rows(output / filename, rows)
        status.update({"status": "complete", "counts": {"sugarcrepe": len(sugar), "pope": len(pope), "transfer": len(transfer)},
            "pope_variant_label_counts": {v: dict(Counter(row["label"] for row in pope if row["variant"] == v)) for v in VARIANTS},
            "selection": {"sugarcrepe": "Seeded uniform subset within each of seven official categories",
                          "pope": "Same seeded 20 image groups across all variants at default size; all original six questions per image retained",
                          "transfer": "Unique SugarCrepe subset images and benchmark-positive captions; benchmark-conditioned, not representative COCO"},
            "images": sorted(records, key=lambda r: r["image"]),
            "manifest_sha256": {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                                for name in ("sugarcrepe.jsonl", "pope.jsonl", "transfer.jsonl")}})
        save_json(status_path, status)
        print(status["counts"], flush=True)
        return status
    except Exception as exc:
        status["status"] = "failed"
        status["error"] = type(exc).__name__ + ": " + str(exc)
        save_json(status_path, status)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/research_v2")
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--sugar-per-type", type=int, default=16)
    parser.add_argument("--pope-questions", type=int, default=120)
    args = parser.parse_args()
    prepare(args.data_dir, args.seed, args.sugar_per_type, args.pope_questions)
