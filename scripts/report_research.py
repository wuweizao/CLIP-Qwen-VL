"""Regenerate evidence-backed research-v2 reports (including partial progress)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mmresearch.research_common import load_config
from mmresearch.research_report import run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/research.json")
    args = parser.parse_args()
    run(load_config(args.config))
