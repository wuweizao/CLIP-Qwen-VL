"""Prepare the immutable second-stage confirmation dataset."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mmresearch.common import config
from mmresearch.research_data import prepare


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/research.json")
    args = parser.parse_args()
    source = prepare(config(args.config))
    print(source["counts"])
