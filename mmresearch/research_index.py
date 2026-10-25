import numpy as np
from .common import save_json, read_json, save_rows
from .research_common import paths, load_split, fingerprint, context_path
from .retrieval import Retriever
from .metrics import recall_at_k


def run(cfg):
    from .contexts import build_contexts, match_context_lengths, diversity_subset
    data, out, reports = paths(cfg)
    out.mkdir(parents=True, exist_ok=True)
    train = load_split(cfg, "train")
    model = Retriever(cfg)
    images = {}
    for split in ("train", "validation", "test"):
        rows = load_split(cfg, split)
        images[split] = model.encode([r["image"] for r in rows], images=True)
    captions = [c for r in train for c in r["captions"]]
    owners = [r["id"] for r in train for _ in r["captions"]]
    caption_features = model.encode(captions)
    np.savez_compressed(out / "index.npz", **images, caption_features=caption_features,
                        caption_owners=np.array(owners), captions=np.array(captions))
    from .data import training_subset
    random_ids = [r["id"] for r in training_subset(train, cfg["core_train_images"] / len(train), cfg["data_seed"])]
    diverse_ids = diversity_subset([r["id"] for r in train], images["train"], cfg["core_train_images"], cfg["data_seed"])
    selection_path = out / "selections.json"
    selections = read_json(selection_path) if selection_path.exists() else {}
    for key, values in (("random", random_ids), ("diversity", diverse_ids)):
        if key in selections and selections[key] != values:
            raise ValueError("Refusing to replace frozen sample IDs; use a new study output directory")
        selections[key] = values
    save_json(selection_path, selections)
    for split in ("train", "validation", "test"):
        rows = load_split(cfg, split)
        contexts = build_contexts(rows, train, images[split], images["train"], caption_features,
                                  owners, seed=cfg["data_seed"], k=cfg["rag_k"], max_words=cfg["context_max_words"])
        contexts = match_context_lengths(contexts)
        save_json(context_path(cfg, split), contexts)
        if split != "train":
            indices = [i for i, r in enumerate(train) if r["id"] in set(random_ids)]
            bank = [train[i] for i in indices]
            keep = [i for i, owner in enumerate(owners) if owner in set(random_ids)]
            subset = build_contexts(rows, bank, images[split], images["train"][indices], caption_features[keep],
                                    [owners[i] for i in keep], seed=cfg["data_seed"], k=cfg["rag_k"], max_words=cfg["context_max_words"])
            save_json(context_path(cfg, split, "subset"), match_context_lengths(subset))
    test = load_split(cfg, "test")
    texts = [c for r in test for c in r["captions"]]
    test_owners = [i for i, r in enumerate(test) for _ in r["captions"]]
    features = model.encode(texts)
    similarity = images["test"] @ features.T
    result = {"images": len(test), "captions": len(texts),
              "image_to_text": recall_at_k(similarity, [[j for j, owner in enumerate(test_owners) if owner == i] for i in range(len(test))]),
              "text_to_image": recall_at_k(similarity.T, [[i] for i in test_owners]),
              "identity": fingerprint(cfg)}
    save_json(out / "retrieval_metrics.json", result)
    save_json(out / "index_identity.json", fingerprint(cfg))
    # The semantic selector encodes a fixed researcher vocabulary on CPU;
    # it never reads the candidate images' human captions to choose labels.
    from scripts.refresh_research_sampling import run as refresh_semantic_sampling
    refresh_semantic_sampling(cfg)
    print("Research index prepared:", result, flush=True)
