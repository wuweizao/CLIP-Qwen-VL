"""Resumable research plan. Each GPU stage exits before the next starts."""
import argparse
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mmresearch.common import ROOT, read_json, save_json
from mmresearch.research_common import load_config, paths, fingerprint


def plan(cfg, phase):
    steps = []
    if phase in ("all", "core"):
        steps.append(["index"])
        steps.append(["evaluate", "--regime", "zero_shot", "--seed", "42", "--full"])
        for seed in cfg["seeds"]:
            for regime in cfg["regimes"]:
                options = ["--regime", regime, "--seed", str(seed)]
                steps.extend([["train", *options], ["evaluate", *options, "--full"], ["report"]])
    if phase in ("all", "aux"):
        for seed in cfg["seeds"]:
            for sampling in ("diversity", "coverage"):
                options = ["--regime", "plain", "--seed", str(seed), "--sampling", sampling]
                steps.extend([["train", *options], ["evaluate", *options]])
            for budget in ("epochs", "matched_steps"):
                options = ["--regime", "plain", "--seed", str(seed), "--count", "32", "--budget", budget]
                steps.extend([["train", *options], ["evaluate", *options]])
        for budget in ("single_epoch", "matched_steps"):
            options = ["--regime", "plain", "--seed", "42", "--count", str(cfg["train_images"]), "--budget", budget]
            steps.extend([["train", *options], ["evaluate", *options]])
        options = ["--regime", "plain", "--seed", "42", "--scope", "language_merger"]
        steps.extend([["train", *options], ["evaluate", *options]])
        for regime in ("zero_shot", *cfg["regimes"]):
            steps.append(["bench", "--regime", regime, "--seed", "42"])
        steps.extend([["human"], ["report"], ["verify"]])
    return steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/research.json")
    parser.add_argument("--phase", choices=["all", "core", "aux"], default="all")
    args = parser.parse_args()
    cfg = load_config(args.config)
    _, out, _ = paths(cfg)
    out.mkdir(parents=True, exist_ok=True)
    tasks = plan(cfg, args.phase)
    identity = fingerprint(cfg)
    path = out / ("execution_" + args.phase + ".json")
    if path.exists():
        state = read_json(path)
        if state["identity"] != identity or [r["args"] for r in state["steps"]] != tasks:
            raise ValueError("Cannot resume a changed experiment plan")
    else:
        state = {"identity": identity, "phase": args.phase,
                 "steps": [{"args": task, "status": "pending"} for task in tasks]}
    save_json(path, state)
    for i, step in enumerate(state["steps"]):
        if step["status"] == "completed":
            continue
        print("Research step %d/%d: %s" % (i + 1, len(tasks), " ".join(step["args"])), flush=True)
        step["status"] = "running"
        save_json(path, state)
        start = time.perf_counter()
        result = subprocess.run([sys.executable, "scripts/research.py", *step["args"], "--config", args.config], cwd=ROOT)
        step.update(status="completed" if result.returncode == 0 else "failed",
                    seconds=time.perf_counter() - start, exit_code=result.returncode)
        save_json(path, state)
        if result.returncode:
            raise SystemExit(result.returncode)
    print("Research plan completed:", args.phase, flush=True)


if __name__ == "__main__":
    main()
