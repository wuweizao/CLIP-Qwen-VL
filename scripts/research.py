"""Entry point for the controlled second-stage research experiments."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mmresearch.research_common import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["index", "train", "evaluate", "report", "bench", "human", "import-human", "verify"])
    parser.add_argument("--config", default="configs/research.json")
    parser.add_argument("--regime", default="plain", choices=["zero_shot", "plain", "retrieval", "noise"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling", default="random", choices=["random", "diversity", "coverage"])
    parser.add_argument("--count", type=int)
    parser.add_argument("--scope", default="language", choices=["language", "language_merger"])
    parser.add_argument("--budget", default="epochs", choices=["epochs", "matched_steps", "single_epoch"])
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--scores", help="Path to a CSV actually completed by a human reviewer")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.command == "index":
        from mmresearch.research_index import run
        run(cfg)
    elif args.command == "train":
        from mmresearch.research_train import run
        run(cfg, args.regime, args.seed, args.sampling, args.count, args.scope, args.budget)
    elif args.command == "evaluate":
        from mmresearch.research_evaluate import run_bundle
        run_bundle(cfg, args.regime, args.seed, args.sampling, args.count, args.scope, args.budget, args.full)
    elif args.command == "report":
        from mmresearch.research_report import run
        run(cfg)
    elif args.command == "bench":
        from mmresearch.research_bench import run
        run(cfg, args.regime, args.seed)
    elif args.command == "human":
        from mmresearch.human_review import run
        run(cfg)
    elif args.command == "import-human":
        if not args.scores:
            parser.error("--scores is required")
        from mmresearch.human_review import ingest_human_scores
        print(ingest_human_scores(cfg, args.scores))
    else:
        from mmresearch.research_verify import run
        run(cfg)


if __name__ == "__main__":
    main()
