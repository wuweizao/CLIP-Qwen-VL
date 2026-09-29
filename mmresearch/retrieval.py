import numpy as np
from .common import ROOT, read_rows, save_json, provenance
from .metrics import recall_at_k


class Retriever:
    def __init__(self, cfg):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(cfg["clip_model"]).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(cfg["clip_model"])
        self.batch_size = cfg["batch_size"]

    def encode(self, values, images=False):
        import torch
        from PIL import Image
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(values), self.batch_size):
                batch = values[start:start + self.batch_size]
                if images:
                    batch = [Image.open(ROOT / p).convert("RGB") for p in batch]
                    inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                    features = self.model.get_image_features(**inputs)
                else:
                    inputs = self.processor(text=batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
                    features = self.model.get_text_features(**inputs)
                outputs.append(torch.nn.functional.normalize(features.float(), dim=-1).cpu().numpy())
        return np.concatenate(outputs)


def run(cfg):
    model = Retriever(cfg)
    train = read_rows(ROOT / "data/train.jsonl")
    test = read_rows(ROOT / "data/test.jsonl")
    texts, owners = [], []
    for i, row in enumerate(test):
        texts.extend(row["captions"])
        owners.extend([i] * len(row["captions"]))
    image_features = model.encode([r["image"] for r in test], images=True)
    text_features = model.encode(texts)
    scores = image_features @ text_features.T
    result = {"image_to_text": recall_at_k(scores, [[j for j, owner in enumerate(owners) if owner == i] for i in range(len(test))]),
              "text_to_image": recall_at_k(scores.T, [[owner] for owner in owners]),
              "test_images": len(test), "test_captions": len(texts), "model": cfg["clip_model"]}
    save_json(ROOT / "outputs/retrieval.json", result)
    bank = model.encode([r["image"] for r in train], images=True)
    for split in ("validation", "test"):
        rows = read_rows(ROOT / "data" / (split + ".jsonl"))
        if set(r["id"] for r in rows) & set(r["id"] for r in train):
            raise ValueError("Train/evaluation overlap")
        features = image_features if split == "test" else model.encode([r["image"] for r in rows], images=True)
        ranking = np.argsort(-(features @ bank.T), axis=1, kind="stable")
        context = {r["id"]: [{"id": train[j]["id"], "caption": train[j]["captions"][0]} for j in order[:cfg["rag_k"]]]
                   for r, order in zip(rows, ranking)}
        save_json(ROOT / "outputs" / (split + "_contexts.json"), context)
    np.savez_compressed(ROOT / "outputs/retrieval_index.npz", images=image_features, texts=text_features,
                        bank=bank, image_ids=np.array([r["id"] for r in test]), captions=np.array(texts))
    save_json(ROOT / "outputs/retrieval_provenance.json", provenance(cfg))
    return result
