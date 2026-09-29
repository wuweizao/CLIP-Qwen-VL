import argparse
from .common import config


def main():
    parser = argparse.ArgumentParser(description="CLIP + Qwen-VL research pipeline")
    parser.add_argument("command", choices=["prepare", "retrieve", "train", "evaluate", "report", "infer"])
    parser.add_argument("--config", default="configs/small.json")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--adapter")
    parser.add_argument("--prompt", choices=["concise", "grounded"], default="concise")
    parser.add_argument("--rag", action="store_true")
    parser.add_argument("--split", choices=["test", "validation"], default="test")
    parser.add_argument("--image")
    args = parser.parse_args()
    cfg = config(args.config)
    if args.command == "prepare":
        from .data import prepare
        prepare(cfg)
    elif args.command == "retrieve":
        from .retrieval import run
        print(run(cfg))
    elif args.command == "train":
        from .train import run
        run(cfg, args.fraction)
    elif args.command == "evaluate":
        from .evaluate import run
        print(run(cfg, args.adapter, args.prompt, args.rag, args.split))
    elif args.command == "report":
        from .report import run
        run()
    else:
        if not args.image:
            parser.error("--image is required for infer")
        import torch
        from .vl import load, inputs
        model, processor = load(cfg, args.adapter)
        model.eval()
        batch = inputs(processor, [{"id": "query", "image": args.image}], args.prompt).to(model.device)
        with torch.inference_mode():
            output = model.generate(**batch, do_sample=False, max_new_tokens=cfg["max_new_tokens"])
        print(processor.decode(output[0, batch["input_ids"].shape[1]:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
