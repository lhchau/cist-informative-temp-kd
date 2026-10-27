"""Mean +- std ROUGE-L over seeds for every evaluated checkpoint (the format of Table 4).

    python tools/collect_results.py results/gpt2/*/seed10/best_rougeL
"""
import ast
import glob
import os
import sys

import numpy as np

BENCHMARKS = ["dolly", "vicuna", "sinst", "uinst"]


def rouge_l(log_path):
    """ROUGE-L from the last 'test |' line written by evaluate.py."""
    with open(log_path) as f:
        line = [l for l in f if l.startswith("test |")][-1]
    return ast.literal_eval(line.split(" | ")[2])["rougeL"]


for ckpt in sys.argv[1:]:
    cells, means = [], []
    for bench in BENCHMARKS:
        scores = [rouge_l(p) for p in sorted(glob.glob(os.path.join(ckpt, "eval", bench, "seed*", "log.txt")))]
        if scores:
            means.append(np.mean(scores))
            cells.append(f"{bench} {np.mean(scores):.2f}±{np.std(scores):.2f} (n={len(scores)})")
        else:
            cells.append(f"{bench} -")
    avg = f"{np.mean(means):.2f}" if len(means) == len(BENCHMARKS) else "-"
    print(f"{ckpt}: " + " | ".join(cells) + f" | avg {avg}")
