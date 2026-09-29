"""Run each stage in a separate process so GPU memory is released."""
import subprocess
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mmresearch.common import ROOT, config


def run(*args):
    print("Running:", " ".join(args), flush=True)
    subprocess.run([sys.executable, "-m", "mmresearch.cli", *args], cwd=ROOT, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", action="store_true", help="Reuse existing immutable data manifests")
    args = parser.parse_args()
    cfg = config()
    if not args.prepared:
        run("prepare")
    run("retrieve")
    for adapter in [None] + ["outputs/sft_%g/adapter" % f for f in cfg["fractions"]]:
        if adapter:
            run("train", "--fraction", adapter.split("/")[1].removeprefix("sft_"))
        for prompt in cfg["prompts"]:
            for rag in (False, True):
                args = ["evaluate", "--prompt", prompt]
                if adapter:
                    args += ["--adapter", adapter]
                if rag:
                    args += ["--rag"]
                run(*args)
    run("report")
    subprocess.run([sys.executable, "scripts/verify_run.py"], cwd=ROOT, check=True)
