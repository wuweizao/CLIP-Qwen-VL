"""Supplementary inference-bank controls, independent from the main plan."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mmresearch.research_common import load_config
from mmresearch.research_aux_controls import run, summarize


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/research.json")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    result = (summarize if args.summarize_only else run)(load_config(args.config))
    print("Supplementary bank control:", result["status"], result["completed_evaluations"], "/", result["expected_evaluations"])
