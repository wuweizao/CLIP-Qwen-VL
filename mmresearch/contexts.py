"""Deterministic, leakage-safe retrieval interventions for captioning research.

The query captions are never inspected. ``score`` is cosine similarity (or the
documented reranking average), not a calibrated confidence probability. Synthetic
perturbations are lexical interventions, not verified statements about the image.
"""
import copy
import hashlib
import math
import random
import re
from collections import Counter

import numpy as np


MODES = ("none", "random", "image", "text", "rerank", "partial", "conflict")
_REPLACEMENTS = {
    "dog": "cat", "dogs": "cats", "cat": "dog", "cats": "dogs",
    "horse": "cow", "horses": "cows", "cow": "horse", "cows": "horses",
    "bird": "dog", "birds": "dogs", "red": "blue", "blue": "red",
    "green": "yellow", "yellow": "green", "white": "black", "black": "white",
    "one": "two", "two": "three", "three": "one", "four": "two",
    "1": "2", "2": "3", "3": "1", "4": "2", "left": "right",
    "right": "left", "above": "below", "below": "above", "behind": "beside",
}
_COVERAGE = set(_REPLACEMENTS) | {
    "person", "people", "man", "woman", "boy", "girl", "child", "children",
    "baby", "car", "boat", "bicycle", "ball", "water", "snow", "tree",
    "running", "walking", "sitting", "standing", "swimming", "riding",
    "jumping", "holding", "wearing", "chasing", "near", "under", "over",
}


def _seed(seed, key):
    value = hashlib.sha256((str(seed) + "|" + str(key)).encode("utf-8")).digest()
    return int.from_bytes(value[:8], "big")


def _features(values, rows, label):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) != rows or not np.isfinite(values).all():
        raise ValueError("Invalid " + label + " feature matrix")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Zero-norm " + label + " embedding")
    return values / norms


def _cap_caption(text, max_words):
    text = " ".join(str(text).split())
    original_words = len(text.split())
    if not original_words:
        raise ValueError("Bank captions must be nonempty")
    capped = text
    complete = True
    if original_words > max_words:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        selected = []
        for sentence in sentences:
            if len((" ".join(selected + [sentence])).split()) > max_words:
                break
            selected.append(sentence)
        if selected:
            capped = " ".join(selected)
        else:
            capped = " ".join(text.split()[:max_words]).rstrip(" ,;:.!?") + "."
            complete = False
    return capped, {"original_words": original_words, "effective_words": len(capped.split()),
                    "truncated": capped != text, "complete_sentence_boundary": complete}


def _perturb(text):
    changes = []
    def replace(match):
        word = match.group(0)
        value = _REPLACEMENTS.get(word.lower())
        if value is None or changes:
            return word
        if word[0].isupper():
            value = value.capitalize()
        changes.append({"from": word, "to": value})
        return value
    return re.sub(r"\b(?:[A-Za-z]+|[0-9]+)\b", replace, text), changes


def build_contexts(query_rows, bank_rows, query_features, bank_image_features,
                   bank_caption_features, caption_owners, seed=42, k=2, max_words=20):
    """Build matched-source interventions without inspecting query references.

    ``caption_owners`` accepts bank row indices or IDs. For each owner, caption
    feature occurrences correspond to that row's captions in list order. Text
    and rerank retrieval select the best caption per image, then distinct images.
    Image retrieval uses each selected image's first caption, matching baseline.
    ``partial`` combines ceil(k/2) top images with deterministic random decoys.
    Returns ``{mode: {query_id: [records]}}``. Fewer than k eligible bank images
    yield fewer records; same-owner images and every associated caption are
    excluded even when querying the training bank itself.
    """
    if k < 1 or max_words < 1:
        raise ValueError("k and max_words must be positive")
    bank_ids = [r["id"] for r in bank_rows]
    query_ids = [r["id"] for r in query_rows]
    if len(set(bank_ids)) != len(bank_ids) or len(set(query_ids)) != len(query_ids):
        raise ValueError("Bank and query IDs must each be unique")
    if any(not r.get("captions") for r in bank_rows):
        raise ValueError("Each bank image needs at least one caption")
    query = _features(query_features, len(query_rows), "query")
    bank = _features(bank_image_features, len(bank_rows), "bank image")
    captions = _features(bank_caption_features, len(caption_owners), "bank caption")
    if query.shape[1] != bank.shape[1] or query.shape[1] != captions.shape[1]:
        raise ValueError("Embedding dimensions differ")
    id_to_index = {value: i for i, value in enumerate(bank_ids)}
    owner_indices, caption_values = [], []
    occurrences = Counter()
    for owner in caption_owners:
        if isinstance(owner, (int, np.integer)) and 0 <= int(owner) < len(bank_rows):
            owner_index = int(owner)
        elif owner in id_to_index:
            owner_index = id_to_index[owner]
        else:
            raise ValueError("Unknown caption owner")
        position = occurrences[owner_index]
        if position >= len(bank_rows[owner_index]["captions"]):
            raise ValueError("More caption embeddings than owner captions")
        occurrences[owner_index] += 1
        owner_indices.append(owner_index)
        caption_values.append(bank_rows[owner_index]["captions"][position])
    owner_indices = np.asarray(owner_indices, dtype=int)
    positions_by_owner = [np.flatnonzero(owner_indices == owner) for owner in range(len(bank_rows))]
    result = {mode: {} for mode in MODES}
    for i, query_id in enumerate(query_ids):
        image_scores = bank @ query[i]
        text_scores = captions @ query[i]
        eligible = [j for j, value in enumerate(bank_ids) if value != query_id]
        eligible.sort(key=lambda j: str(bank_ids[j]))
        image_order = sorted(eligible, key=lambda j: (-image_scores[j], str(bank_ids[j])))
        image_margin = float(image_scores[image_order[0]] - image_scores[image_order[1]]) if len(image_order) > 1 else 0.0

        def record(owner, caption, score, source, margin=image_margin, **extra):
            capped, metadata = _cap_caption(caption, max_words)
            return {"id": bank_ids[owner], "caption": capped, "score": float(score),
                    "margin": margin, "source": source, **metadata, **extra}

        result["none"][query_id] = []
        result["image"][query_id] = [record(j, bank_rows[j]["captions"][0], image_scores[j], "image") for j in image_order[:k]]
        shuffled = eligible.copy()
        random.Random(_seed(seed, query_id)).shuffle(shuffled)
        result["random"][query_id] = [record(j, bank_rows[j]["captions"][0], image_scores[j], "random") for j in shuffled[:k]]
        relevant_count = min(len(image_order), (k + 1) // 2)
        partial = image_order[:relevant_count]
        partial += [j for j in shuffled if j not in partial][:max(0, k - len(partial))]
        result["partial"][query_id] = [record(j, bank_rows[j]["captions"][0], image_scores[j], "partial", relevance_part="top" if p < relevant_count else "random") for p, j in enumerate(partial)]

        best_by_owner = {}
        for owner in eligible:
            positions = positions_by_owner[owner]
            if len(positions):
                best_by_owner[owner] = min(positions, key=lambda p: (-text_scores[p], str(caption_values[p]), int(p)))
        for mode in ("text", "rerank"):
            scored = []
            for owner in eligible:
                if owner not in best_by_owner:
                    continue
                # The image score is constant for this owner's captions.
                best = best_by_owner[owner]
                score = text_scores[best] if mode == "text" else .5 * (text_scores[best] + image_scores[owner])
                scored.append((float(score), owner, int(best)))
            scored.sort(key=lambda value: (-value[0], str(bank_ids[value[1]])))
            margin = scored[0][0] - scored[1][0] if len(scored) > 1 else 0.0
            result[mode][query_id] = [record(owner, caption_values[p], score, mode, margin=margin,
                image_score=float(image_scores[owner]), text_score=float(text_scores[p])) for score, owner, p in scored[:k]]

        conflict = []
        for original in result["image"][query_id]:
            item = copy.deepcopy(original)
            item["caption"], changes = _perturb(item["caption"])
            item.update(source="conflict", synthetic=True, verified_false=False,
                        perturbations=changes, perturbation_applied=bool(changes))
            item["effective_words"] = len(item["caption"].split())
            conflict.append(item)
        result["conflict"][query_id] = conflict
    return result


def match_context_lengths(contexts_by_mode):
    """Return a copy with exact per-slot word matching across nonempty modes.

    No padding text is introduced. Each slot is trimmed to the shortest source
    caption for that query. Empty conditions remain empty. If a nonempty mode
    contains fewer items, all nonempty modes are reduced to that count. Call
    before adaptive gating so abstention is not treated as a matching condition.
    """
    result = copy.deepcopy(contexts_by_mode)
    query_ids = set().union(*(set(v) for v in result.values())) if result else set()
    for query_id in query_ids:
        groups = [values[query_id] for values in result.values() if values.get(query_id)]
        if not groups:
            continue
        count = min(map(len, groups))
        for values in result.values():
            if values.get(query_id):
                del values[query_id][count:]
        for slot in range(count):
            target = min(len(group[slot]["caption"].split()) for group in groups)
            for group in groups:
                item = group[slot]
                words = item["caption"].split()
                item["pre_match_words"] = len(words)
                item["length_matched"] = True
                item["matched_words"] = target
                if len(words) > target:
                    item["caption"] = " ".join(words[:target]).rstrip(" ,;:.!?") + "."
                    item["truncated"] = True
                    item["complete_sentence_boundary"] = False
                item["effective_words"] = len(item["caption"].split())
    return result


def training_context(row_id, contexts, regime, seed=42):
    """Select plain, image-retrieval, or uniform noise-mixture training context.

    Noise mixture choices are image/partial/random/none, stable for (seed,id).
    Caller may vary seed by epoch for deliberately resampled training context.
    """
    if regime == "plain":
        return []
    if regime == "retrieval":
        mode = "image"
    elif regime == "noise":
        mode = random.Random(_seed(seed, "training:" + str(row_id))).choice(("image", "partial", "random", "none"))
    else:
        raise ValueError("Unknown training regime: " + str(regime))
    if row_id not in contexts.get(mode, {}):
        raise KeyError("Missing training query: " + str(row_id))
    values = copy.deepcopy(contexts[mode][row_id])
    if any(item["id"] == row_id for item in values):
        raise ValueError("Training context contains current image")
    for item in values:
        item["training_mode"] = mode
    return values


def select_context(contexts, min_score, min_margin=0.0, k=2):
    """Gate query-level retrieval using top score and top-two margin.

    Thresholds must be selected outside this function using validation data only.
    The decision emits no context or up to k supplied contexts; no labels enter.
    """
    if k < 1 or not math.isfinite(min_score) or not math.isfinite(min_margin):
        raise ValueError("Invalid gate settings")
    if not contexts:
        return []
    first = contexts[0]
    if first["score"] < min_score or first.get("margin", 0.0) < min_margin:
        return []
    return copy.deepcopy(contexts[:k])


def diversity_subset(ids, features, count, seed=42):
    """Farthest-point sampling on unit embeddings, stable under input reorder."""
    if len(set(ids)) != len(ids) or not 0 <= count <= len(ids):
        raise ValueError("Unique IDs and count in [0,n] required")
    values = _features(features, len(ids), "sampling")
    if count == 0:
        return []
    order = sorted(range(len(ids)), key=lambda j: str(ids[j]))
    first = random.Random(_seed(seed, "diversity")).choice(order)
    selected = [first]
    similarity = values @ values[first]
    while len(selected) < count:
        remaining = [j for j in order if j not in selected]
        chosen = min(remaining, key=lambda j: (similarity[j], str(ids[j])))
        selected.append(chosen)
        similarity = np.maximum(similarity, values @ values[chosen])
    return [ids[j] for j in selected]


def coverage_subset(rows, count, seed=42):
    """Greedy coverage of visible-category/action keywords in training captions.

    This is caption-vocabulary coverage, not verified object/relationship labels.
    It intentionally consumes bank captions only and cannot establish factual
    coverage without annotations. Rare keywords receive inverse-frequency weight.
    """
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids) or not 0 <= count <= len(rows):
        raise ValueError("Unique IDs and count in [0,n] required")
    labels = [set(re.findall(r"[a-z0-9]+", " ".join(r["captions"]).lower())) & _COVERAGE for r in rows]
    frequencies = Counter(word for group in labels for word in group)
    order = sorted(range(len(rows)), key=lambda j: str(ids[j]))
    random.Random(_seed(seed, "coverage")).shuffle(order)
    tie_order = {index: rank for rank, index in enumerate(order)}
    selected, covered = [], set()
    for _ in range(count):
        candidates = [j for j in order if j not in selected]
        chosen = min(candidates, key=lambda j: (-sum(1 / frequencies[w] for w in labels[j] - covered), tie_order[j]))
        selected.append(chosen)
        covered.update(labels[chosen])
    return [ids[j] for j in selected]
