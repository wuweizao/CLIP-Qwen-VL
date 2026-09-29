"""Search the evaluated gallery using arbitrary text or an image."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from mmresearch.common import ROOT, config, read_rows
from mmresearch.retrieval import Retriever

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--text")
group.add_argument("--image")
parser.add_argument("--top-k", type=int, default=5)
args = parser.parse_args()
if args.top_k < 1:
    parser.error("--top-k must be positive")
index = np.load(ROOT / "outputs/retrieval_index.npz")
model = Retriever(config())
if args.text:
    scores = model.encode([args.text])[0] @ index["images"].T
    rows = read_rows(ROOT / "data/test.jsonl")
    values = [r["image"] for r in rows]
else:
    scores = model.encode([args.image], images=True)[0] @ index["texts"].T
    values = index["captions"].tolist()
for rank, i in enumerate(np.argsort(-scores)[:args.top_k], 1):
    print("%d\t%.4f\t%s" % (rank, scores[i], values[i]))
