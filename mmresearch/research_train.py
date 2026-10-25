"""Controlled LoRA training with plain, retrieval-aware and mixed-noise prompts."""
import collections
import math
import time
from .common import ROOT, read_json, save_json, seed_all, digest
from .research_common import paths, load_split, fingerprint, run_name, context_path, assert_completed
from .vl import load, messages


class ContextCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        import torch
        from PIL import Image
        if len(examples) != 1:
            raise ValueError("Microbatch size must be 1")
        row = examples[0]
        image = Image.open(ROOT / row["image"]).convert("RGB")
        context = row.get("context") or None
        prefix = self.processor.apply_chat_template(messages(row, context=context), tokenize=False, add_generation_prompt=True)
        full = self.processor.apply_chat_template(messages(row, context=context, answer=row["answer"]), tokenize=False, add_generation_prompt=False)
        batch = self.processor(text=[full], images=[image], return_tensors="pt")
        prefix_ids = self.processor(text=[prefix], images=[image], return_tensors="pt")["input_ids"]
        n = prefix_ids.shape[1]
        if not torch.equal(batch["input_ids"][:, :n], prefix_ids):
            raise ValueError("Assistant prefix token alignment failed")
        labels = batch["input_ids"].clone()
        labels[:, :n] = -100
        labels[batch["attention_mask"] == 0] = -100
        if not (labels != -100).any():
            raise ValueError("Missing assistant tokens")
        batch["labels"] = labels
        batch["_image_id"] = row["id"]
        return batch


def run(cfg, regime="plain", seed=42, sampling="random", count=None, scope="language", budget="epochs"):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments
    from .contexts import training_context
    data, out, _ = paths(cfg)
    count = count or cfg["core_train_images"]
    name = run_name(regime, seed, sampling, count, scope, budget)
    directory = out / "models" / name
    identity = {**fingerprint(cfg), "regime": regime, "seed": seed, "sampling": sampling,
                "count": count, "scope": scope, "budget": budget,
                "training_code_sha256": digest(__file__),
                "contexts_sha256": digest(context_path(cfg, "train"))}
    if assert_completed(directory, identity):
        print("Reusing completed adapter:", name, flush=True)
        return directory / "adapter"
    train_all = load_split(cfg, "train")
    from .data import training_subset
    selections = read_json(out / "selections.json")
    if sampling == "random":
        selected = [r["id"] for r in training_subset(train_all, count / len(train_all), cfg["data_seed"])]
    else:
        selected = selections[sampling][:count]
        if len(selected) != count:
            raise ValueError("Insufficient precomputed selected images")
    contexts = read_json(context_path(cfg, "train"))
    train = []
    for row in train_all:
        if row["id"] not in set(selected):
            continue
        context = training_context(row["id"], contexts, regime, cfg["data_seed"])
        if any(c["id"] == row["id"] for c in context):
            raise ValueError("Self-caption leakage during training")
        for caption in row["captions"][:cfg["captions_per_train_image"]]:
            train.append({**row, "answer": caption, "context": context})
    validation_contexts = read_json(context_path(cfg, "validation"))
    # Same validation objective (no context) for every training regime.
    selection_rows = load_split(cfg, "validation")[cfg["gate_validation_images"]:]
    if not selection_rows:
        raise ValueError("Checkpoint selection and retrieval gate need disjoint validation subsets")
    valid = [{**r, "answer": r["captions"][0], "context": []} for r in selection_rows]
    seed_all(seed)
    model, processor = load(cfg, training=True)
    model.config.use_cache = False
    lora = LoraConfig(r=cfg["lora_rank"], lora_alpha=2 * cfg["lora_rank"], lora_dropout=.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                      modules_to_save=["merger"] if scope == "language_merger" else None,
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.enable_input_require_grads()
    seen = collections.Counter()

    class AuditedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            image_id = inputs.pop("_image_id")
            if model.training:
                seen[image_id] += 1
            return super().compute_loss(model, inputs, return_outputs=return_outputs,
                                        num_items_in_batch=num_items_in_batch)

    core_steps = math.ceil(cfg["core_train_images"] * cfg["captions_per_train_image"] / cfg["gradient_accumulation_steps"]) * cfg["epochs"]
    epochs = 1 if budget == "single_epoch" else cfg["epochs"]
    args = TrainingArguments(output_dir=str(directory), num_train_epochs=epochs,
        max_steps=core_steps if budget == "matched_steps" else -1,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"], learning_rate=cfg["learning_rate"],
        bf16=torch.cuda.is_available(), gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, logging_steps=4,
        eval_strategy="steps", eval_steps=core_steps if budget == "matched_steps" else max(1, math.ceil(len(train) / cfg["gradient_accumulation_steps"])),
        save_strategy="steps", save_steps=core_steps if budget == "matched_steps" else max(1, math.ceil(len(train) / cfg["gradient_accumulation_steps"])),
        save_total_limit=1, save_only_model=True, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False, remove_unused_columns=False,
        report_to="none", dataloader_num_workers=0, seed=seed, data_seed=seed, optim="adamw_torch")
    trainer = AuditedTrainer(model=model, args=args, train_dataset=train, eval_dataset=valid,
                             data_collator=ContextCollator(processor))
    start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trainer.train()
    trainer.save_model(str(directory / "adapter"))
    processor.save_pretrained(str(directory / "adapter"))
    save_json(directory / "history.json", trainer.state.log_history)
    save_json(directory / "training.json", {"identity": identity, "selected_image_ids": selected,
        "selected_instructions": len(train), "optimizer_steps": trainer.state.global_step,
        "checkpoint_validation_ids": [r["id"] for r in selection_rows],
        "unique_supervised_images": len(seen), "supervised_microbatches": sum(seen.values()),
        "context_modes": dict(collections.Counter(c.get("source", "unknown") for r in train for c in r["context"])),
        "seconds": time.perf_counter() - start, "best_checkpoint": trainer.state.best_model_checkpoint,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else None})
    save_json(directory / "completed.json", {"identity": identity})
    print("Completed adapter:", name, flush=True)
    return directory / "adapter"
