"""Freeze label-free semantic coverage using existing image vectors and CPU CLIP.

Only selections.json is updated. Random/diversity selections and existing core
models are preserved. This script never places tensors or a model on the GPU.
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from mmresearch.common import digest, read_json, save_json
from mmresearch.research_common import load_config, load_split, paths
from mmresearch.research_sampling import TAG_PROMPTS, semantic_coverage


def run(cfg, batch_size=32, threads=4):
    import torch
    from transformers import CLIPModel, CLIPProcessor
    if batch_size < 1 or threads < 1:
        raise ValueError("Positive CPU batch size and thread count required")
    torch.set_num_threads(threads)
    data, out, _ = paths(cfg)
    selection_path = out / "selections.json"
    selections = read_json(selection_path)
    preserved = {key: list(selections[key]) for key in ("random", "diversity")}
    rows = load_split(cfg, "train")
    ids = [row["id"] for row in rows]
    index_identity = read_json(out / "index_identity.json")
    if index_identity["data_manifests"]["train"] != digest(data / "train.jsonl"):
        raise ValueError("Cached training embeddings do not match the immutable image manifest")
    with np.load(out / "index.npz") as index:
        image_features = index["train"].copy()
    if len(image_features) != len(ids):
        raise ValueError("Training features and image ID count differ")
    model = CLIPModel.from_pretrained(cfg["clip_model"], local_files_only=True,
                                      torch_dtype=torch.float32).to("cpu").eval()
    processor = CLIPProcessor.from_pretrained(cfg["clip_model"], local_files_only=True)
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise RuntimeError("Sampling refresh must run on CPU only")
    features = []
    with torch.inference_mode():
        for start in range(0, len(TAG_PROMPTS), batch_size):
            batch = processor(text=TAG_PROMPTS[start:start + batch_size], padding=True,
                              truncation=True, return_tensors="pt")
            values = model.get_text_features(**batch)
            features.append(torch.nn.functional.normalize(values.float(), dim=-1).numpy())
    tag_features = np.concatenate(features)
    count = int(cfg["core_train_images"])
    selected = semantic_coverage(ids, image_features, tag_features, count, seed=cfg["data_seed"])
    policy = {"name": "semantic_features_only", "algorithm": "top3_zscore_tag_coverage_then_farthest_point",
              "selection_inputs": ["unlabeled_training_image_embeddings", "fixed_researcher_tag_embeddings"],
              "human_captions_used_for_selection": False, "evaluation_labels_used_for_selection": False,
              "tag_prompts": TAG_PROMPTS, "tag_count": len(TAG_PROMPTS), "top_tags_per_image": 3,
              "seed": cfg["data_seed"], "count": count, "selected_ids": selected,
              "train_manifest_sha256": digest(data / "train.jsonl"),
              "image_feature_sha256": hashlib.sha256(image_features.tobytes()).hexdigest(),
              "tag_feature_sha256": hashlib.sha256(tag_features.tobytes()).hexdigest(),
              "sampling_code_sha256": digest(Path(__file__).resolve().parents[1] / "mmresearch/research_sampling.py"),
              "clip_model": cfg["clip_model"], "device": "cpu", "frozen": True,
              "interpretation": "CLIP semantic heuristics, not verified object/action/attribute annotations"}
    completed = list((out / "models").glob("*_coverage_*/completed.json"))
    previous_policy = selections.get("_policies", {}).get("coverage")
    if completed and (selections.get("coverage") != selected or previous_policy != policy):
        raise ValueError("Coverage models already exist; refusing to change their frozen selection policy")
    selections["coverage"] = selected
    selections.setdefault("_policies", {})["coverage"] = policy
    if any(selections[key] != preserved[key] for key in preserved):
        raise AssertionError("Random/diversity selection changed")
    save_json(selection_path, selections)
    print("Frozen semantic_features_only coverage: %d/%d images, %d predefined tags, CPU only" %
          (len(selected), len(ids), len(TAG_PROMPTS)), flush=True)
    return selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/research.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    run(load_config(args.config), args.batch_size, args.threads)
