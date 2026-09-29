"""Dependency-light, explicitly defined metrics. All scores use [0, 1]."""
import re
from collections import Counter


def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def token_f1(prediction, reference):
    p, r = tokens(prediction), tokens(reference)
    if not p or not r:
        return float(p == r)
    overlap = sum((Counter(p) & Counter(r)).values())
    return 2 * overlap / (len(p) + len(r))


def rouge_l(prediction, reference):
    p, r = tokens(prediction), tokens(reference)
    if not p or not r:
        return float(p == r)
    prev = [0] * (len(r) + 1)
    for a in p:
        cur = [0]
        for j, b in enumerate(r):
            cur.append(prev[j] + 1 if a == b else max(prev[j + 1], cur[-1]))
        prev = cur
    return 2 * prev[-1] / (len(p) + len(r))


def caption_scores(predictions, references):
    if len(predictions) != len(references) or not predictions or any(not r for r in references):
        raise ValueError("Nonempty predictions and aligned reference groups required")
    return {name: sum(max(fn(p, r) for r in refs) for p, refs in zip(predictions, references)) / len(predictions)
            for name, fn in (("token_f1", token_f1), ("rouge_l", rouge_l))}


def recall_at_k(scores, positives, ks=(1, 5, 10)):
    import numpy as np
    scores = np.asarray(scores)
    if scores.ndim != 2 or len(positives) != len(scores) or not len(scores) or scores.shape[1] == 0:
        raise ValueError("Invalid retrieval matrix")
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite retrieval scores")
    if any(not p or any(i < 0 or i >= scores.shape[1] for i in p) for p in positives):
        raise ValueError("Every query needs valid positive indices")
    ranking = np.argsort(-scores, axis=1, kind="stable")
    return {"R@%s" % k: float(np.mean([bool(set(row[:min(k, len(row))]) & set(pos))
                                       for row, pos in zip(ranking, positives)])) for k in ks}


def paired_bootstrap(first, second, seed=42, samples=2000):
    """Mean(second-first), percentile CI resampling image-level paired scores."""
    import numpy as np
    if len(first) != len(second) or not first:
        raise ValueError("Aligned nonempty scores required")
    delta = np.asarray(second) - np.asarray(first)
    rng = np.random.default_rng(seed)
    means = delta[rng.integers(0, len(delta), size=(samples, len(delta)))].mean(axis=1)
    return {"delta": float(delta.mean()), "ci_low": float(np.quantile(means, .025)),
            "ci_high": float(np.quantile(means, .975))}
