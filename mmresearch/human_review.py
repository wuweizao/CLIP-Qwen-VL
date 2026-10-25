"""Export blinded local review material; never manufacture human annotations.

Images are embedded only in ignored local outputs. Public reports contain status
counts and instructions, never redistributed dataset pixels or model-labelled
review HTML. Annotation imports are explicit user-supplied evidence; this module
does not authenticate the annotator or substitute automatic metrics for labels.
"""
import base64
import csv
import hashlib
import html
import io
import random
from pathlib import Path

from .common import ROOT, digest, read_json, read_rows, save_json
from .research_common import paths, load_split, run_name


_CAPTION_FIELDS = ["blind_id", "image_id", "factual_correctness", "detail_coverage", "notes", "reviewer"]
_RELEVANCE_FIELDS = ["review_id", "query_image_id", "bank_image_id", "relevance", "notes", "reviewer"]


def _identifier(*values):
    return hashlib.sha256("|".join(str(value) for value in values).encode("utf-8")).hexdigest()[:16]


def _image_uri(relative):
    from PIL import Image
    path = (ROOT / relative).resolve()
    data = (ROOT / "data").resolve()
    if data not in path.parents:
        raise ValueError("Review images must come from project data/")
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((480, 320))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=78)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _write_csv(path, fieldnames, rows, identifier):
    old = {}
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as handle:
            old = {row[identifier]: row for row in csv.DictReader(handle)}
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            previous = old.get(row[identifier], {})
            writer.writerow({field: previous.get(field, row.get(field, "")) for field in fieldnames})


def _document(title, introduction, body):
    return """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>%s</title><style>
body{max-width:1080px;margin:32px auto;padding:0 20px;font:16px/1.6 system-ui;color:#172033;background:#f5f7fa}
article{background:white;border:1px solid #dbe1ea;padding:20px;border-radius:12px;margin:20px 0}
img{max-width:100%%;height:auto;border-radius:8px}code{font-size:13px}h1{font-size:26px}
.candidate{padding:12px 16px;margin:12px 0;background:#f1f5f9;border-radius:8px}.pair{display:flex;gap:24px;flex-wrap:wrap}
.pair>div{flex:1;min-width:240px}a{color:#1768b8}.warning{background:#fff3cd;padding:12px}
</style><h1>%s</h1>%s%s</html>""" % (html.escape(title), html.escape(title), introduction, body)


def _status(cfg, mapping, annotation_dir, missing):
    _, out, reports = paths(cfg)
    annotations = read_json(annotation_dir / "annotations.json") if (annotation_dir / "annotations.json").exists() else {"captions": {}, "relevance": {}}
    caption_ids = set(mapping["captions"])
    relevance_ids = set(mapping["relevance"])
    caption_scored = len(caption_ids & set(annotations.get("captions", {})))
    relevance_scored = len(relevance_ids & set(annotations.get("relevance", {})))
    ready = not missing and bool(caption_ids) and bool(relevance_ids)
    complete = ready and caption_scored == len(caption_ids) and relevance_scored == len(relevance_ids)
    result = {"package_status": "ready" if ready else "partial_materials", "human_scoring_completed": complete,
              "caption_items_ready": len(caption_ids), "caption_items_scored": caption_scored,
              "caption_items_unscored": len(caption_ids) - caption_scored,
              "relevance_items_ready": len(relevance_ids), "relevance_items_scored": relevance_scored,
              "relevance_items_unscored": len(relevance_ids) - relevance_scored,
              "missing_materials": missing,
              "local_files": ["review.html", "blind_scores.csv", "relevance_review.html", "relevance_scores.csv"],
              "instructions": ["Open the ignored outputs human_review folder locally; images are not distributed in public reports.",
                "Score captions while viewing the actual image, without opening mapping.json or model identities.",
                "Caption scores are integers 0,1,2; use both factual_correctness and detail_coverage columns.",
                "Relevance labels are relevant, partial, or irrelevant; leave uncertain items blank and add a note.",
                "Import edited CSV files explicitly. Blank templates and automatic metrics are not human judgments.",
                "Human-provided CSV provenance is recorded but annotator identity is not independently authenticated."],
              "automatic_fact_labels_created": False}
    save_json(reports / "human_review_status.json", result)
    return result


def run(cfg):
    """Export first 32 confirmation images, with stable per-image blind ordering.

    Conditions are zero-shot / plain SFT / noise SFT, first configured training
    seed, concise no-context outputs. Only predictions actually on disk enter the
    package; unavailable conditions are listed as pending, not filled in.
    """
    _, out, _ = paths(cfg)
    directory = out / "human_review"
    directory.mkdir(parents=True, exist_ok=True)
    rows = load_split(cfg, "test")[:32]
    train = {row["id"]: row for row in load_split(cfg, "train")}
    seed = cfg["seeds"][0]
    names = ["zero_shot"] + [run_name(regime, seed, "random", cfg["core_train_images"], "language", "epochs")
                             for regime in ("plain", "noise")]
    predictions = {}
    missing = []
    for name in names:
        path = out / "evaluations" / name / "test_none_full/predictions.jsonl"
        if not path.exists():
            missing.append("predictions:" + name)
            continue
        values = read_rows(path)
        if len({row["id"] for row in values}) != len(values):
            raise ValueError("Duplicate prediction IDs in human review source")
        predictions[name] = {row["id"]: row for row in values}
    mapping = {"schema_version": 1, "seed": seed, "image_ids": [row["id"] for row in rows],
               "captions": {}, "relevance": {}, "label_source": "pending_human_annotation"}
    score_rows, sections = [], []
    for row in rows:
        entries = []
        for name in names:
            prediction = predictions.get(name, {}).get(row["id"])
            if prediction is None:
                if name in predictions:
                    missing.append("prediction:" + name + ":" + row["id"])
                continue
            text = prediction["prediction"]
            identifier = "c_" + _identifier(seed, row["id"], name, hashlib.sha256(text.encode()).hexdigest())
            entries.append((identifier, text))
            mapping["captions"][identifier] = {"image_id": row["id"], "model": name,
                "prediction_sha256": hashlib.sha256(text.encode()).hexdigest()}
            score_rows.append({"blind_id": identifier, "image_id": row["id"]})
        random.Random(int(_identifier(seed, row["id"]), 16)).shuffle(entries)
        if entries:
            candidates = "".join('<div class="candidate"><code>%s</code><p>%s</p></div>' %
                                  (identifier, html.escape(text)) for identifier, text in entries)
            sections.append('<article><p>图像 <code>%s</code></p><img alt="待评估图像" src="%s">%s</article>' %
                            (html.escape(row["id"]), _image_uri(row["image"]), candidates))
    introduction = '<p class="warning">这些条目尚未经过人工判断。请勿查看模型映射文件。</p><p>在 <a href="blind_scores.csv">blind_scores.csv</a> 填写评分：事实正确性 0=多数不受图像支持，1=有错误或不确定细节，2=描述均有图像支持；细节覆盖率 0=遗漏大部分主要信息，1=覆盖部分，2=覆盖主要对象、动作和显著细节。两项都填 0–2 的整数，不确定时留空并备注。</p>'
    (directory / "review.html").write_text(_document("图像描述人工盲评", introduction, "".join(sections)), encoding="utf-8")
    # CSV order must be blind as well; otherwise condition-major ordering leaks identities.
    random.Random(int(_identifier(seed, "caption_csv"), 16)).shuffle(score_rows)
    _write_csv(directory / "blind_scores.csv", _CAPTION_FIELDS, score_rows, "blind_id")

    context_file = out / "contexts/test_full.json"
    contexts = read_json(context_file).get("image", {}) if context_file.exists() else {}
    if not context_file.exists():
        missing.append("image_retrieval_contexts")
    relevance_rows, sections = [], []
    for row in rows:
        items = contexts.get(row["id"], [])
        if context_file.exists() and not items:
            missing.append("retrieval_context:" + row["id"])
        cards = []
        for rank, item in enumerate(items):
            if item["id"] not in train:
                raise ValueError("Review retrieval image is outside training bank")
            if item["id"] == row["id"]:
                raise ValueError("Self-caption leakage in review package")
            identifier = "r_" + _identifier(seed, "relevance", row["id"], item["id"], rank, item["caption"])
            mapping["relevance"][identifier] = {"query_image_id": row["id"], "bank_image_id": item["id"],
                "rank": rank + 1, "caption": item["caption"], "source": "image_retrieval"}
            relevance_rows.append({"review_id": identifier, "query_image_id": row["id"], "bank_image_id": item["id"]})
            cards.append('<div class="candidate"><code>%s</code><div class="pair"><div><p>查询图像</p><img alt="查询图像" src="%s"></div><div><p>检索图像</p><img alt="检索图像" src="%s"></div></div><p>检索描述：%s</p></div>' %
                         (identifier, _image_uri(row["image"]), _image_uri(train[item["id"]]["image"]), html.escape(item["caption"])))
        if cards:
            sections.append("<article>" + "".join(cards) + "</article>")
    introduction = '<p class="warning">未自动认定任何描述相关或错误。相似图像不等于描述适用于查询图像。</p><p>在 <a href="relevance_scores.csv">relevance_scores.csv</a> 的 relevance 列填写 relevant（主要内容适用于查询图像）、partial（部分适用，含不匹配细节）或 irrelevant（主要内容不适用）。查看查询图像再判断，无法确定时留空并备注。</p>'
    (directory / "relevance_review.html").write_text(_document("检索描述相关性人工核验", introduction, "".join(sections)), encoding="utf-8")
    _write_csv(directory / "relevance_scores.csv", _RELEVANCE_FIELDS, relevance_rows, "review_id")
    mapping["missing_materials"] = sorted(set(missing))
    save_json(directory / "mapping.json", mapping)
    status = _status(cfg, mapping, directory, sorted(set(missing)))
    print("Human review materials:", status, flush=True)
    return status


def ingest_human_scores(cfg, path):
    """Import explicitly edited caption or relevance CSV with strict ID checks.

    Partial annotation is allowed, but at least one fully scored item is needed.
    Existing accepted annotations are retained. No model-generated scorer runs.
    """
    _, out, _ = paths(cfg)
    directory = out / "human_review"
    mapping = read_json(directory / "mapping.json")
    source = Path(path)
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    kind = "captions" if "blind_id" in fieldnames else "relevance" if "review_id" in fieldnames else None
    if kind is None:
        raise ValueError("Expected blind_id or review_id column")
    identifier_key = "blind_id" if kind == "captions" else "review_id"
    accepted, seen = {}, set()
    for row in rows:
        identifier = row.get(identifier_key, "").strip()
        if identifier not in mapping[kind] or identifier in seen:
            raise ValueError("Unknown or duplicate blind review ID: " + identifier)
        seen.add(identifier)
        if kind == "captions":
            a = row.get("factual_correctness", "").strip()
            b = row.get("detail_coverage", "").strip()
            if not a and not b:
                continue
            if a not in ("0", "1", "2") or b not in ("0", "1", "2"):
                raise ValueError("Both caption scores must be integer 0, 1, or 2")
            values = {"factual_correctness": int(a), "detail_coverage": int(b)}
        else:
            label = row.get("relevance", "").strip().lower()
            if not label:
                continue
            if label not in ("relevant", "partial", "irrelevant"):
                raise ValueError("Relevance must be relevant, partial, or irrelevant")
            values = {"relevance": label}
        accepted[identifier] = {**values, "notes": row.get("notes", "").strip(),
            "reviewer": row.get("reviewer", "").strip(), "label_source": "human_provided",
            "source_csv_sha256": digest(source), "annotator_identity_authenticated": False}
    if not accepted:
        raise ValueError("No scored items: blank templates are not human annotations")
    annotation_path = directory / "annotations.json"
    annotations = read_json(annotation_path) if annotation_path.exists() else {"captions": {}, "relevance": {}}
    annotations[kind].update(accepted)
    save_json(annotation_path, annotations)
    grouped = {}
    for identifier, values in annotations["captions"].items():
        if identifier not in mapping["captions"]:
            continue
        name = mapping["captions"][identifier]["model"]
        grouped.setdefault(name, []).append(values)
    summary = {name: {"n": len(values), **{key: sum(row[key] for row in values) / len(values)
               for key in ("factual_correctness", "detail_coverage")}} for name, values in grouped.items()}
    save_json(directory / "human_summary.json", {"label_source": "human_provided", "caption_scores": summary,
              "relevance_counts": {label: sum(row["relevance"] == label for row in annotations["relevance"].values())
                                   for label in ("relevant", "partial", "irrelevant")},
              "warning": "May be incomplete, unpaired, and from one unverified annotator; not an automatic factuality score."})
    status = _status(cfg, mapping, directory, mapping.get("missing_materials", []))
    print("Imported %d human-provided %s judgments" % (len(accepted), kind), flush=True)
    return status
