"""Feature-only semantic-coverage sampling with a fixed tag vocabulary.

No human caption, object annotation, evaluation label, or model prediction is
accepted by the selector. Tags are researcher-specified prompts frozen before
auxiliary sampling experiments. CLIP tag similarities are semantic heuristics,
not verified object/action/attribute labels.
"""
import hashlib
import random

import numpy as np


# Fixed a priori vocabulary; never generated from the candidate pool captions.
TAG_PROMPTS = [
    "A photo of a person.", "A photo of a man.", "A photo of a woman.",
    "A photo of a child.", "A photo of a baby.", "A photo of several people.",
    "A photo of a dog.", "A photo of a cat.", "A photo of a horse.",
    "A photo of a bird.", "A photo of a cow.", "A photo of a sheep.",
    "A photo of a wild animal.", "A photo of a car.", "A photo of a bicycle.",
    "A photo of a motorcycle.", "A photo of a boat.", "A photo of a train.",
    "A photo of a bus.", "A photo of an airplane.", "A photo of a ball.",
    "A photo of a frisbee.", "A photo of a bench.", "A photo of an umbrella.",
    "A photo of a tree.", "A photo of flowers.", "A photo of a building.",
    "A photo of a bridge.", "A photo of a mountain.", "A photo of a beach.",
    "A photo of water.", "A photo of snow.", "A photo of grass.",
    "A photo of a street.", "A photo of a playground.", "A photo of a hat.",
    "A photo of a helmet.", "A photo of a bag.", "A photo of a red object.",
    "A photo of a blue object.", "A photo of a green object.",
    "A photo of a yellow object.", "A photo of a white object.",
    "A photo of a black object.", "A photo of an orange object.",
    "A photo of a brown object.", "A photo of a pink object.",
    "A photo of a purple object.", "A photo of someone running.",
    "A photo of someone walking.", "A photo of someone sitting.",
    "A photo of someone standing.", "A photo of someone jumping.",
    "A photo of someone swimming.", "A photo of someone riding a horse.",
    "A photo of someone riding a bicycle.", "A photo of someone climbing.",
    "A photo of someone skiing.", "A photo of someone playing a sport.",
    "A photo of someone carrying an object.", "A photo of someone holding an object.",
    "A photo of someone eating.", "A photo of someone drinking.",
    "A photo of someone sleeping.", "A photo of people rowing a boat.",
    "A photo of someone catching a ball.", "A photo of someone looking at something.",
    "A photo of people embracing.", "A photo of an animal running.",
    "A photo of an animal jumping.", "A photo of an animal swimming.",
    "A photo of a person beside an animal.", "A photo of a person in front of a vehicle.",
    "A photo of a person behind a vehicle.", "A photo of a child on a bicycle.",
    "A photo of a person under an umbrella.", "A photo of an object above another object.",
    "A photo of an object below another object.", "A photo of a person next to another person.",
    "A photo of an animal chasing another animal.", "A photo of a person walking a dog.",
    "A photo of an animal in water.", "A photo of a person on snow.",
    "A photo of a person inside a building.", "A photo of a person outdoors.",
    "A photo of a crowd in a city.", "A photo of a person in a field.",
    "A photo of a person near a tree.", "A photo of an object to the left of another object.",
    "A photo of an object to the right of another object.",
]


def _normalized(features, label):
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or not len(features) or not features.shape[1] or not np.isfinite(features).all():
        raise ValueError("Invalid " + label + " features")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Zero-norm " + label + " features")
    return features / norms


def semantic_coverage(ids, image_features, tag_features, count, seed=42):
    """Select IDs by fixed-tag coverage, breaking ties by visual diversity.

    Similarities are standardized separately for each tag across the available
    image pool. Each image is associated with its top three standardized tags.
    Greedy selection maximizes newly covered tags; ties prefer the image furthest
    from the selected set under cosine distance. Once tag coverage saturates,
    this becomes farthest-point sampling. Final ties use a seeded permutation of
    canonically sorted IDs. Returned selections are stable under input reordering.

    This uses unlabeled image embeddings and predefined text tag embeddings only.
    Similarity-derived tag assignments are not object or relation ground truth.
    """
    ids = list(ids)
    if len(ids) != len(set(ids)) or not isinstance(count, (int, np.integer)) or not 0 <= count <= len(ids):
        raise ValueError("Unique IDs and integer count in [0,n] required")
    images = _normalized(image_features, "image")
    tags = _normalized(tag_features, "tag")
    if len(images) != len(ids) or images.shape[1] != tags.shape[1]:
        raise ValueError("Image IDs or feature dimensions are misaligned")
    if count == 0:
        return []
    order = sorted(range(len(ids)), key=lambda index: str(ids[index]))
    canonical_ids = [ids[index] for index in order]
    images = images[order]
    similarities = images @ tags.T
    spread = similarities.std(axis=0)
    standardized = (similarities - similarities.mean(axis=0)) / np.where(spread > 1e-12, spread, 1.)
    top = np.argsort(-standardized, axis=1, kind="stable")[:, :min(3, len(tags))]
    tag_sets = [set(map(int, row)) for row in top]
    tie_order = list(range(len(ids)))
    seed_value = int.from_bytes(hashlib.sha256((str(seed) + "|semantic_coverage").encode()).digest()[:8], "big")
    random.Random(seed_value).shuffle(tie_order)
    tie_rank = {value: rank for rank, value in enumerate(tie_order)}
    selected, covered = [], set()
    remaining = set(range(len(ids)))
    max_similarity = np.full(len(ids), -1.)
    for _ in range(int(count)):
        def priority(index):
            uncovered = len(tag_sets[index] - covered)
            distance = 1. - max_similarity[index] if selected else 0.
            return (-uncovered, -round(float(distance), 12), tie_rank[index])
        chosen = min(remaining, key=priority)
        selected.append(chosen)
        remaining.remove(chosen)
        covered.update(tag_sets[chosen])
        max_similarity = np.maximum(max_similarity, images @ images[chosen])
    return [canonical_ids[index] for index in selected]
