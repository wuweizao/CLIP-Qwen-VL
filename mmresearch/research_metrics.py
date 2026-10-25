"""Metrics for the deeper experimental suite, with explicit scoring protocols.

CIDEr follows the n-gram TF-IDF cosine consensus formulation of Vedantam et al.
(CVPR 2015), https://arxiv.org/abs/1411.5726. CIDEr-D additionally applies the
reference-count clipping and Gaussian length penalty of the COCO evaluator.
Unlike the usual COCO evaluation protocol, document frequencies here are fitted
once to supplied training/validation reference groups and then held fixed. The
tokenizer is the project's lowercase ASCII word/number tokenizer, not PTB.
Scores therefore must be labeled ``CIDEr_fixed_df_regex`` / ``CIDErD_fixed_df_regex``
and must not be presented as directly comparable COCO leaderboard scores.

Lexical context overlap and reference coverage below are diagnostic proxies.
They do not detect unsupported visual facts and are not hallucination metrics.
"""

import math
import re
from collections import Counter

from .metrics import rouge_l, token_f1, tokens

_STOPWORDS = frozenset(
    "a an the this that these those is are was were be been being am do does did "
    "have has had and or but if as at by for from in into of on onto to with "
    "without over under through near along around up down out off it its they "
    "their them he him his she her we our you your i my there here some any "
    "one two three four five six seven eight nine ten very more most than then "
    "while who which what where when how not no yes".split()
)


def _reference_groups(groups, name):
    if not groups or any(not isinstance(g, (list, tuple)) or not g for g in groups):
        raise ValueError(name + " must contain nonempty reference groups")
    if any(not isinstance(caption, str) for group in groups for caption in group):
        raise ValueError(name + " must contain strings")


def _ngrams(text, n=4):
    words = tokens(text)
    return [Counter(tuple(words[i:i + order]) for i in range(len(words) - order + 1))
            for order in range(1, n + 1)]


class FixedDFCider:
    """Reusable scorer with document frequencies frozen before test scoring.

    Each image/reference group is one document, irrespective of how many
    captions it has. Unseen n-grams receive df=1, matching COCO's max(1, df)
    convention. Empty or zero-IDF n-gram vectors contribute zero for that order.
    ``d_variant=False`` is CIDEr; True adds clipping and the length penalty.
    """

    def __init__(self, document_frequency_references, n=4, d_variant=False, sigma=6.0):
        _reference_groups(document_frequency_references, "document_frequency_references")
        if len(document_frequency_references) < 2:
            raise ValueError("At least two fixed DF documents are required")
        if n < 1 or sigma <= 0:
            raise ValueError("n and sigma must be positive")
        self.n = n
        self.d_variant = d_variant
        self.sigma = float(sigma)
        self.documents = len(document_frequency_references)
        self.document_frequency = Counter()
        for group in document_frequency_references:
            observed = set()
            for reference in group:
                for order_counts in _ngrams(reference, n):
                    observed.update(order_counts)
            self.document_frequency.update(observed)

    @property
    def protocol(self):
        return {"name": "CIDErD_fixed_df_regex" if self.d_variant else "CIDEr_fixed_df_regex",
                "df_documents": self.documents, "ngram_orders": self.n,
                "idf": "log(N / max(1, image_document_frequency))",
                "tokenizer": "lowercase_[a-z0-9]+", "scale": 10.0,
                "sigma": self.sigma if self.d_variant else None}

    def _vectors(self, text):
        vectors = []
        norms = []
        for counts in _ngrams(text, self.n):
            vector = {gram: count * math.log(self.documents / max(1, self.document_frequency[gram]))
                      for gram, count in counts.items()}
            vectors.append(vector)
            norms.append(math.sqrt(sum(value * value for value in vector.values())))
        return vectors, norms

    def score(self, prediction, references):
        if not isinstance(prediction, str) or not references or any(not isinstance(r, str) for r in references):
            raise ValueError("Prediction string and nonempty string references required")
        pred_vectors, pred_norms = self._vectors(prediction)
        total = 0.0
        for reference in references:
            ref_vectors, ref_norms = self._vectors(reference)
            similarity = 0.0
            for hyp, ref, hyp_norm, ref_norm in zip(pred_vectors, ref_vectors, pred_norms, ref_norms):
                if hyp_norm == 0 or ref_norm == 0:
                    continue
                if self.d_variant:
                    dot = sum(min(weight, ref.get(gram, 0.0)) * ref.get(gram, 0.0)
                              for gram, weight in hyp.items())
                else:
                    dot = sum(weight * ref.get(gram, 0.0) for gram, weight in hyp.items())
                similarity += dot / (hyp_norm * ref_norm)
            if self.d_variant:
                delta = len(tokens(prediction)) - len(tokens(reference))
                similarity *= math.exp(-(delta * delta) / (2 * self.sigma * self.sigma))
            total += similarity / self.n
        return float(10.0 * total / len(references))

    def score_many(self, predictions, references):
        if len(predictions) != len(references) or not predictions:
            raise ValueError("Aligned nonempty predictions and references required")
        _reference_groups(references, "references")
        return [self.score(prediction, refs) for prediction, refs in zip(predictions, references)]


def cider_scores(predictions, references, document_frequency_references):
    """Return per-image CIDEr scores under the fixed-DF regex protocol."""
    return FixedDFCider(document_frequency_references).score_many(predictions, references)


def cider_d_scores(predictions, references, document_frequency_references):
    """Return per-image clipped/length-penalized CIDEr-D, fixed-DF regex protocol."""
    return FixedDFCider(document_frequency_references, d_variant=True).score_many(predictions, references)


def _content_words(text):
    return [word for word in tokens(text) if word not in _STOPWORDS]


def per_sample_metrics(prediction, references, context=None):
    """Text diagnostics; context-copy/coverage values are lexical, not factual.

    reference_detail_recall is maximum content-token recall against one
    reference. It is explicitly a lexical detail-coverage proxy, not a measure
    of all visible details. Empty content references contribute zero.
    """
    if not references:
        raise ValueError("At least one reference is required")
    content = _content_words(prediction)
    context_text = []
    for item in context or []:
        context_text.append(item if isinstance(item, str) else item["caption"])
    vocabulary = set(_content_words(" ".join(context_text)))
    pred_counts = Counter(content)
    coverage = []
    for reference in references:
        ref_words = _content_words(reference)
        coverage.append(sum((pred_counts & Counter(ref_words)).values()) / len(ref_words) if ref_words else 0.0)
    return {"token_f1": max(token_f1(prediction, reference) for reference in references),
            "rouge_l": max(rouge_l(prediction, reference) for reference in references),
            "words": len(tokens(prediction)),
            "context_copy_rate": sum(word in vocabulary for word in content) / len(content) if content else 0.0,
            "reference_detail_recall": max(coverage)}


def _row_score(row, metric):
    value = row[metric] if metric in row else row["metrics"][metric]
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Non-finite metric")
    return value


def _aligned(rows_a, rows_b):
    a, b = {}, {}
    for rows, mapping in ((rows_a, a), (rows_b, b)):
        for row in rows:
            if row["id"] in mapping:
                raise ValueError("Duplicate image ID")
            mapping[row["id"]] = row
    if not a or set(a) != set(b):
        raise ValueError("Identical nonempty image ID sets are required")
    ids = sorted(a)
    return ids, a, b


def paired_effect(rows_a, rows_b, metric="token_f1", seed=42, samples=2000):
    """Mean B-A and image-paired bootstrap CI for ONE fixed training run.

    This CI describes evaluation-image variation. It does not describe
    training-seed variation; report ``seed_summary`` separately for that.
    """
    import numpy as np
    if samples < 1:
        raise ValueError("samples must be positive")
    ids, a, b = _aligned(rows_a, rows_b)
    delta = np.asarray([_row_score(b[i], metric) - _row_score(a[i], metric) for i in ids])
    rng = np.random.default_rng(seed)
    means = []
    # Bounded temporary memory for larger test sets / more bootstrap draws.
    for start in range(0, samples, 128):
        indices = rng.integers(0, len(ids), size=(min(128, samples - start), len(ids)))
        means.extend(delta[indices].mean(axis=1).tolist())
    return {"metric": metric, "delta": float(delta.mean()),
            "ci_low": float(np.quantile(means, .025)), "ci_high": float(np.quantile(means, .975)),
            "n_images": len(ids), "bootstrap_samples": samples, "bootstrap_seed": seed,
            "uncertainty_unit": "evaluation_image_conditional_on_fixed_training_run"}


def seed_summary(seed_results, metric="token_f1"):
    """Mean and sample SD of complete run scores, one independent training seed each.

    Entries must have unique ``seed`` values and the metric at top level.
    No training-seed confidence interval is claimed for a small number of runs.
    """
    import numpy as np
    if not seed_results or len({r["seed"] for r in seed_results}) != len(seed_results):
        raise ValueError("One nonempty result per unique training seed required")
    values = np.asarray([_row_score(row, metric) for row in seed_results])
    return {"metric": metric, "mean": float(values.mean()),
            "seed_sd": float(values.std(ddof=1)) if len(values) > 1 else None,
            "n_training_seeds": len(values), "seeds": [r["seed"] for r in seed_results],
            "uncertainty_unit": "training_seed"}


def paired_seed_effect(seed_rows_a, seed_rows_b, metric="token_f1"):
    """Aggregate paired per-seed mean differences; avoids pooling seeds as images."""
    if set(seed_rows_a) != set(seed_rows_b) or not seed_rows_a:
        raise ValueError("Matching nonempty training seed sets required")
    results = []
    for seed in sorted(seed_rows_a):
        ids, a, b = _aligned(seed_rows_a[seed], seed_rows_b[seed])
        results.append({"seed": seed, "delta": sum(_row_score(b[i], metric) - _row_score(a[i], metric) for i in ids) / len(ids)})
    return {**seed_summary(results, metric="delta"), "source_metric": metric, "per_seed": results}


def answer_yes_no(text):
    """Strict leading answer parser returning yes, no, or invalid.

    Optional ``Answer:`` and simple Markdown emphasis are accepted. Embedded
    yes/no words in an explanation are never searched to manufacture a label.
    A standalone answer line may precede an explanation on subsequent lines.
    """
    if not isinstance(text, str):
        return "invalid"
    cleaned = text.strip()
    cleaned = re.sub(r"^answer\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.lstrip("*_` ")
    match = re.match(r"^(yes|no)\b", cleaned, flags=re.IGNORECASE)
    if not match:
        return "invalid"
    label = match.group(1).lower()
    suffix = cleaned[match.end():].lstrip("*_` ")
    # Reject 'yes/no', 'no or yes', and unpunctuated speculative answers.
    if not suffix or suffix.startswith((".", ",", "!", "\n", "\r", ":", ";")):
        return label
    return "invalid"


def binary_metrics(predictions, labels, require_balanced=False):
    """Binary object-presence metrics with invalid outputs counted as errors.

    Invalid positive examples contribute FN; invalid negative examples remain
    incorrect in accuracy and in macro-class F1. Precision/recall/F1 fields
    refer to the positive class. ``invalid_rate`` must always be reported.
    Labels must already be explicit yes/no labels; inputs are model output
    strings parsed with the frozen ``answer_yes_no`` protocol above.
    """
    if not predictions or len(predictions) != len(labels) or any(x not in ("yes", "no") for x in labels):
        raise ValueError("Aligned nonempty predictions and yes/no labels required")
    counts = Counter(labels)
    if not counts["yes"] or not counts["no"]:
        raise ValueError("Both positive and negative labels are required")
    if require_balanced and counts["yes"] != counts["no"]:
        raise ValueError("Balanced yes/no labels required by this protocol")
    parsed = [answer_yes_no(p) for p in predictions]
    tp = sum(p == "yes" and y == "yes" for p, y in zip(parsed, labels))
    fp = sum(p == "yes" and y == "no" for p, y in zip(parsed, labels))
    tn = sum(p == "no" and y == "no" for p, y in zip(parsed, labels))
    fn = counts["yes"] - tp
    invalid_positive = sum(p == "invalid" and y == "yes" for p, y in zip(parsed, labels))
    invalid_negative = sum(p == "invalid" and y == "no" for p, y in zip(parsed, labels))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / counts["yes"]
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    negative_precision = tn / sum(p == "no" for p in parsed) if "no" in parsed else 0.0
    negative_recall = tn / counts["no"]
    negative_f1 = 2 * negative_precision * negative_recall / (negative_precision + negative_recall) if negative_precision + negative_recall else 0.0
    return {"n": len(labels), "accuracy": (tp + tn) / len(labels), "precision": precision,
            "recall": recall, "f1": f1, "macro_f1": (f1 + negative_f1) / 2,
            "yes_ratio": parsed.count("yes") / len(parsed),
            "invalid_rate": parsed.count("invalid") / len(parsed),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn, "fn_including_invalid_positive": fn,
            "invalid_positive": invalid_positive, "invalid_negative": invalid_negative,
            "label_counts": dict(counts), "parser": "strict_leading_yes_no_v1"}
