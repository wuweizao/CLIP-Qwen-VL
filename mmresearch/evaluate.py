import time
from .common import ROOT, read_rows, read_json, save_rows, save_json, provenance, seed_all
from .metrics import caption_scores
from .vl import load, inputs


def run(cfg, adapter=None, prompt="concise", rag=False, split="test"):
    import torch
    seed_all(cfg["seed"])
    rows = read_rows(ROOT / "data" / (split + ".jsonl"))
    contexts = read_json(ROOT / "outputs" / (split + "_contexts.json")) if rag else None
    if rag:
        recorded = read_json(ROOT / "outputs/retrieval_provenance.json")
        current = provenance(cfg)
        if recorded["manifests"] != current["manifests"] or recorded["config"]["clip_model"] != cfg["clip_model"]:
            raise ValueError("Stale retrieval cache: rerun retrieve")
    model, processor = load(cfg, adapter)
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    name = ("sft_" + str(adapter).replace("\\", "/").split("/")[-2].removeprefix("sft_") if adapter else "zero_shot")
    out = ROOT / "outputs/evaluations" / ("%s_%s_%s_%s" % (name, prompt, "rag" if rag else "plain", split))
    results = []
    start = time.perf_counter()
    with torch.inference_mode():
        # Small actual batches keep generation within 16GB VRAM.
        for i in range(0, len(rows), 2):
            batch_rows = rows[i:i + 2]
            batch = inputs(processor, batch_rows, prompt, contexts).to(model.device)
            generated = model.generate(**batch, max_new_tokens=cfg["max_new_tokens"], do_sample=False)
            predictions = processor.batch_decode(generated[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)
            results.extend({"id": r["id"], "prediction": p.strip(), "references": r["captions"],
                            "context": (contexts or {}).get(r["id"], [])} for r, p in zip(batch_rows, predictions))
            save_rows(out / "predictions.jsonl", results)
            if len(results) % 16 == 0 or len(results) == len(rows):
                print("%s: %d/%d images" % (out.name, len(results), len(rows)), flush=True)
    metrics = caption_scores([r["prediction"] for r in results], [r["references"] for r in results])
    save_json(out / "metrics.json", {**metrics, "n": len(results), "seconds": time.perf_counter() - start,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else None,
        "model": name, "prompt": prompt, "rag": rag, "split": split, "adapter": str(adapter) if adapter else None})
    save_json(out / "provenance.json", provenance(cfg))
    return metrics
