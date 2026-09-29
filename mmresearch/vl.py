from .common import ROOT

PROMPTS = {
    "concise": "Describe this image in one concise English sentence.",
    "grounded": "Write one English sentence describing the visible main subjects, their actions, and setting. Only mention what the image supports."
}


def messages(row, prompt="concise", context=None, answer=None):
    text = PROMPTS[prompt]
    if context:
        text += "\nCaptions of other visually similar training images (may be irrelevant; do not copy unsupported details):\n"
        text += "\n".join("- " + c["caption"] for c in context)
    result = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
    if answer is not None:
        result.append({"role": "assistant", "content": answer})
    return result


def load(cfg, adapter=None, training=False):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(cfg["vl_model"], min_pixels=56 * 56, max_pixels=cfg["max_pixels"])
    processor.tokenizer.padding_side = "right" if training else "left"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg["vl_model"], torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="sdpa", device_map={"": "cuda:0" if torch.cuda.is_available() else "cpu"})
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    return model, processor


def inputs(processor, rows, prompt="concise", contexts=None):
    from PIL import Image
    chats = [processor.apply_chat_template(messages(r, prompt, (contexts or {}).get(r["id"])),
                                           tokenize=False, add_generation_prompt=True) for r in rows]
    images = [Image.open(ROOT / r["image"]).convert("RGB") for r in rows]
    return processor(text=chats, images=images, padding=True, return_tensors="pt")


class SFTCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        import torch
        from PIL import Image
        # Single-image microbatches avoid variable multimodal prefix alignment.
        if len(examples) != 1:
            raise ValueError("SFT collator requires microbatch size 1")
        row = examples[0]
        image = Image.open(ROOT / row["image"]).convert("RGB")
        prefix = self.processor.apply_chat_template(messages(row), tokenize=False, add_generation_prompt=True)
        full = self.processor.apply_chat_template(messages(row, answer=row["answer"]), tokenize=False, add_generation_prompt=False)
        batch = self.processor(text=[full], images=[image], return_tensors="pt")
        prefix_ids = self.processor(text=[prefix], images=[image], return_tensors="pt")["input_ids"]
        n = prefix_ids.shape[1]
        if not torch.equal(batch["input_ids"][:, :n], prefix_ids):
            raise ValueError("Chat prefix mismatch; refusing to train on incorrect labels")
        labels = batch["input_ids"].clone()
        labels[:, :n] = -100
        labels[batch["attention_mask"] == 0] = -100
        if (labels != -100).sum() == 0:
            raise ValueError("No assistant target tokens")
        batch["labels"] = labels
        return batch
