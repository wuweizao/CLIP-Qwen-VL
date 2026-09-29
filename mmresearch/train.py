from .common import ROOT, read_rows, save_json, seed_all, provenance
from .data import training_subset
from .vl import load, SFTCollator


def run(cfg, fraction=1.0):
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments
    seed_all(cfg["seed"])
    model, processor = load(cfg, training=True)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=cfg["lora_rank"], lora_alpha=2 * cfg["lora_rank"],
                 lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
    model.enable_input_require_grads()
    train_images = training_subset(read_rows(ROOT / "data/train.jsonl"), fraction, cfg["seed"])
    selected = {r["id"] for r in train_images}
    train = [r for r in read_rows(ROOT / "data/instructions.jsonl") if r["image_id"] in selected]
    valid = [{**r, "answer": r["captions"][0]} for r in read_rows(ROOT / "data/validation.jsonl")]
    out = ROOT / "outputs" / ("sft_%g" % fraction)
    args = TrainingArguments(output_dir=str(out), num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"], learning_rate=cfg["learning_rate"],
        bf16=next(model.parameters()).dtype.__str__().endswith("bfloat16"),
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1, eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        remove_unused_columns=False, report_to="none", dataloader_num_workers=0,
        seed=cfg["seed"], data_seed=cfg["seed"], optim="adamw_torch")
    trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=valid,
                      data_collator=SFTCollator(processor))
    trainer.train()
    trainer.save_model(str(out / "adapter"))
    processor.save_pretrained(str(out / "adapter"))
    save_json(out / "history.json", trainer.state.log_history)
    save_json(out / "provenance.json", {**provenance(cfg), "fraction": fraction, "train_images": len(selected),
               "train_instructions": len(train), "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)})
