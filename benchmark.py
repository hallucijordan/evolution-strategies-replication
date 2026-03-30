"""
Benchmark runner — train multiple algorithms on the same environment and
produce a side-by-side learning curve plot.

Usage:
    # Train all registered algorithms, then plot:
    python benchmark.py --env PongNoFrameskip-v4 --algos es random --total_steps 500000

    # Skip training, just re-plot existing results:
    python benchmark.py --env PongNoFrameskip-v4 --algos es random --plot_only
"""

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from utils.logger import load_results


SMOOTHING_WINDOW = 20   # rolling-mean window for noisy curves


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark algorithms on Atari")
    p.add_argument("--env",         default="PongNoFrameskip-v4")
    p.add_argument("--algos",       nargs="+", default=["es", "random"])
    p.add_argument("--total_steps", type=int,  default=500_000)
    p.add_argument("--results_dir", default="results")
    p.add_argument("--plot_only",   action="store_true",
                   help="Skip training; only regenerate the comparison plot")
    return p.parse_args()


def run_training(algo: str, env: str, total_steps: int, results_dir: str):
    cmd = [
        sys.executable, "train.py",
        "--algo",        algo,
        "--env",         env,
        "--total_steps", str(total_steps),
        "--results_dir", results_dir,
    ]
    print(f"\n{'='*60}")
    print(f" Training: {algo}  on  {env}")
    print(f"{'='*60}")
    subprocess.run(cmd, check=True)


def plot_results(algos: list[str], env: str, results_dir: str):
    fig, ax = plt.subplots(figsize=(10, 6))

    for algo in algos:
        run_name = f"{algo}_{env}"
        try:
            df = load_results(results_dir, run_name)
        except FileNotFoundError:
            print(f"[warn] No results found for {run_name} — skipping.")
            continue

        if "mean_return" not in df.columns:
            print(f"[warn] 'mean_return' column missing for {run_name} — skipping.")
            continue

        smoothed = df["mean_return"].rolling(SMOOTHING_WINDOW, min_periods=1).mean()
        ax.plot(df["steps"], smoothed, label=algo, linewidth=1.8)
        ax.fill_between(
            df["steps"],
            df["mean_return"].rolling(SMOOTHING_WINDOW, min_periods=1).min(),
            df["mean_return"].rolling(SMOOTHING_WINDOW, min_periods=1).max(),
            alpha=0.15,
        )

    ax.set_xlabel("Environment Steps", fontsize=12)
    ax.set_ylabel("Mean Episode Return", fontsize=12)
    ax.set_title(f"Algorithm Comparison — {env}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    out_path = Path(results_dir) / f"benchmark_{env}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved → {out_path}")
    plt.show()


def main():
    args = parse_args()

    if not args.plot_only:
        for algo in args.algos:
            run_training(algo, args.env, args.total_steps, args.results_dir)

    plot_results(args.algos, args.env, args.results_dir)


if __name__ == "__main__":
    main()
