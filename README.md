# Evolution Strategies — Replication

A clean, modular replication of **"Evolution Strategies as a Scalable Alternative to Reinforcement Learning"** (Salimans et al., OpenAI, 2017 · [arXiv:1703.03864](https://arxiv.org/abs/1703.03864)).

The pipeline is designed so teammates can easily plug in new algorithms and benchmark them against ES on Atari games.

---

## What's Implemented

| Component | Details |
|-----------|---------|
| **ES algorithm** | Algorithm 1 + 2 from the paper: mirrored sampling, rank-based fitness shaping, weight decay, optional multiprocessing |
| **Atari preprocessing** | Standard chain: no-op reset → frame-skip(4) → 84×84 grayscale → 4-frame stack → reward clipping |
| **CNN policy** | DQN/A3C architecture from Mnih et al. [2016]: 3 conv layers + 2 FC layers |
| **Random baseline** | Uniform random action policy for sanity-checking |
| **Training CLI** | `train.py` — configurable via YAML |
| **Evaluation CLI** | `evaluate.py` — load checkpoint, run real game episodes, optional render |
| **Benchmark runner** | `benchmark.py` — train multiple algos, plot comparison curves |

---

## Project Structure

```
evolution-strategies-replication/
├── algorithms/
│   ├── base.py           # Abstract interface — subclass to add new algorithms
│   ├── es.py             # ES from the paper
│   └── random_search.py  # Random policy baseline
├── envs/
│   └── atari_wrappers.py # Full Atari preprocessing chain
├── models/
│   └── atari_cnn.py      # CNN policy + flat-param helpers for ES
├── utils/
│   ├── fitness_shaping.py # Rank normalization
│   └── logger.py          # CSV training logger
├── configs/
│   ├── es_pong.yaml       # Hyperparameters for Atari
│   └── es_mujoco.yaml     # Hyperparameters for MuJoCo
├── results/               # Training logs and checkpoints (git-ignored)
├── train.py
├── evaluate.py
└── benchmark.py
```

---

## Installation

```bash
# Clone
git clone <repo-url>
cd evolution-strategies-replication

# Install dependencies
pip install -r requirements.txt
```

> **Atari ROMs** are handled automatically by `ale-py` on first run.
> If you hit a ROM error, run: `python -m ale_py.roms`

---

## Quick Start

### Train ES on Pong

```bash
python train.py --algo es --env PongNoFrameskip-v4 --config configs/es_pong.yaml
```

Speed up with parallel rollout workers (recommended — set to your CPU count):

```bash
python train.py --algo es --env PongNoFrameskip-v4 --config configs/es_pong.yaml --n_workers 8
```

### Evaluate a Checkpoint

```bash
# Print per-episode scores
python evaluate.py \
    --env PongNoFrameskip-v4 \
    --checkpoint results/es_PongNoFrameskip-v4/checkpoint.npy

# Watch the agent play in real time
python evaluate.py \
    --env PongNoFrameskip-v4 \
    --checkpoint results/es_PongNoFrameskip-v4/checkpoint.npy \
    --render
```

### Benchmark Multiple Algorithms

```bash
# Train ES + random baseline, then plot comparison
python benchmark.py --env PongNoFrameskip-v4 --algos es random --total_steps 500000

# Re-plot without re-training
python benchmark.py --env PongNoFrameskip-v4 --algos es random --plot_only
```

---

## Configuration

Hyperparameters live in `configs/`. Example (`configs/es_pong.yaml`):

```yaml
algorithm:
  sigma: 0.05          # Perturbation noise std
  lr: 0.01             # Learning rate
  population_size: 100 # Must be even (mirrored sampling)
  weight_decay: 0.005  # L2 regularization
  n_workers: 1         # Set to CPU count for parallel rollouts
  seed: 42
```

Pass any config to `train.py` with `--config path/to/config.yaml`.
Any key under `algorithm:` maps directly to `ESConfig`.

---

## Adding a New Algorithm

1. Create `algorithms/my_algo.py`, subclass `BaseAlgorithm`:

```python
from .base import BaseAlgorithm

class MyAlgo(BaseAlgorithm):
    def train(self, env_fn, total_steps, logger): ...
    def evaluate(self, env, n_episodes=10): ...
    def save(self, path): ...
    def load(self, path): ...
```

2. Register it in `algorithms/__init__.py`:

```python
from .my_algo import MyAlgo

REGISTRY = {
    "es":     ES,
    "random": RandomPolicy,
    "my_algo": MyAlgo,      # ← add here
}
```

3. Run it:

```bash
python train.py --algo my_algo --env PongNoFrameskip-v4
python benchmark.py --algos es my_algo --env PongNoFrameskip-v4
```

---

## ES Algorithm Summary

The update rule per generation (Algorithm 1, Salimans et al.):

```
1. Sample n/2 noise vectors  ε ~ N(0, I)
2. Evaluate n policies:      F(θ + σε)  and  F(θ − σε)   [mirrored sampling]
3. Rank-normalize returns     shaped = rank_normalize([F₁⁺, F₁⁻, ...])
4. Gradient estimate:         g = (1/nσ) · Σᵢ shaped_i · εᵢ
5. Update:                    θ ← θ · (1 − lr·wd) + lr · g
```

Key advantages over policy gradient methods:
- No backpropagation — ~3× less compute per episode
- Workers only communicate scalar returns (not gradients)
- Invariant to action frequency (frame-skip)
- Linear speedup with number of parallel workers

---

## Results Format

Each run produces:

```
results/<algo>_<env>/
├── training_log.csv    # steps, generation, mean_return, max_return, gen_time
└── checkpoint.npy      # flat parameter vector (numpy)
```

Load results programmatically:

```python
from utils.logger import load_results

df = load_results("results", "es_PongNoFrameskip-v4")
print(df[["steps", "mean_return"]].tail())
```

---

## Reference

```bibtex
@article{salimans2017evolution,
  title   = {Evolution Strategies as a Scalable Alternative to Reinforcement Learning},
  author  = {Salimans, Tim and Ho, Jonathan and Chen, Xi and Sidor, Szymon and Sutskever, Ilya},
  journal = {arXiv preprint arXiv:1703.03864},
  year    = {2017}
}
```
