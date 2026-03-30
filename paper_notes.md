# Evolution Strategies as a Scalable Alternative to Reinforcement Learning

**Authors:** Tim Salimans, Jonathan Ho, Xi Chen, Szymon Sidor, Ilya Sutskever (OpenAI)
**arXiv:** 1703.03864v2 — September 7, 2017

---

## Abstract

Explores Evolution Strategies (ES) as a black-box optimization alternative to MDP-based RL (Q-learning, Policy Gradients). Key result: a novel communication strategy using shared random seeds means workers only communicate scalar returns — enabling linear scaling to 1,440+ parallel workers. Solved 3D humanoid walking in 10 minutes; competitive with A3C on Atari after 1 hour.

---

## 1. Introduction

### Key Findings

1. **Virtual batch normalization** and other reparameterizations greatly improve ES reliability on neural network policies.
2. **Highly parallelizable**: linear speedups achieved; 1,440 workers solve MuJoCo 3D Humanoid in under 10 minutes.
3. **Data efficiency**: matches A3C final performance using 3–10x more data, but ~3x less compute (no backprop), and faster wall-clock time.
4. **Better exploration**: ES learns diverse gaits on MuJoCo Humanoid (e.g., walking sideways/backwards) — never observed with TRPO.
5. **Robust**: single set of fixed hyperparameters works across all Atari environments; another fixed set for all MuJoCo environments.

---

## 2. Evolution Strategies

ES is a class of black-box optimization algorithms inspired by natural evolution. Each generation, a population of parameter vectors is perturbed ("mutated") and evaluated by a fitness function. The best are recombined and the process repeats.

This work uses **Natural Evolution Strategies (NES)**: represents the population as a distribution over parameters $p_\psi(\theta)$, then maximizes the average objective $\mathbb{E}_{\theta \sim p_\psi} F(\theta)$ via stochastic gradient ascent.

With an isotropic multivariate Gaussian (mean $\theta$, covariance $\sigma^2 I$):

$$\nabla_\theta \mathbb{E}_{\epsilon \sim \mathcal{N}(0,I)} F(\theta + \sigma\epsilon) = \frac{1}{\sigma} \mathbb{E}_{\epsilon \sim \mathcal{N}(0,I)} \{F(\theta + \sigma\epsilon)\,\epsilon\}$$

### Algorithm 1: Basic ES
```
Input: learning rate α, noise std σ, initial params θ₀
for t = 0, 1, 2, ... do
    Sample ε₁, ..., εₙ ~ N(0, I)
    Compute returns Fᵢ = F(θₜ + σεᵢ) for i = 1,...,n
    Set θₜ₊₁ ← θₜ + α · (1/nσ) · Σᵢ Fᵢεᵢ
end for
```

### Algorithm 2: Parallelized ES
- **n** workers share known random seeds and initial parameters $\theta_0$
- Each worker samples $\epsilon_i \sim \mathcal{N}(0,I)$ and computes return $F_i$
- Workers broadcast only the scalar $F_i$ (not gradients)
- Each worker reconstructs all perturbations using shared seeds and updates $\theta$

**Communication cost**: only scalars per worker per iteration (vs. full gradients in policy gradient methods).

### 2.1 Scaling and Parallelizing ES

Three properties make ES suited for parallelism:
1. Operates on complete episodes — infrequent communication needed
2. Workers only communicate scalar returns (not gradients)
3. No value function approximation required

Episode length is capped at $m$ steps (dynamically adjusted to $2\times$ the mean steps taken), keeping CPU utilization above 50%.

### 2.2 Network Parameterization

**Problem**: random Gaussian parameter perturbations may not explore the action space adequately.

**Solutions**:
- **Atari**: Virtual batch normalization (VBN) — normalizing statistics fixed at training start. Makes policy sensitive to small input changes early in training, ensuring diverse action exploration.
- **MuJoCo**: Discretizing continuous actions into 10 bins encourages broader exploration.

### Additional Tricks
- **Antithetic (mirrored) sampling**: always evaluate pairs $(\epsilon, -\epsilon)$ to reduce variance
- **Fitness shaping**: rank-transform returns before computing parameter updates — removes outlier influence, reduces tendency to fall into local optima
- **Weight decay**: prevents policy network weights from growing too large
- $\sigma$ is treated as a **fixed hyperparameter** (not adapted during training)

---

## 3. Smoothing in Parameter Space vs. Action Space

### Policy Gradient (action-space smoothing)
$$\nabla_\theta F_{PG}(\theta) = \mathbb{E}_\epsilon \{ R(\mathbf{a}(\epsilon,\theta)) \nabla_\theta \log p(\mathbf{a};\theta) \}$$

### Evolution Strategies (parameter-space smoothing)
$$\nabla_\theta F_{ES}(\theta) = \mathbb{E}_\xi \{ R(\mathbf{a}(\xi,\theta)) \nabla_\theta \log p(\tilde{\theta}(\xi,\theta);\theta) \}$$

### 3.1 When is ES Better Than Policy Gradients?

Gradient variance comparison:
- **PG**: $\text{Var}[\nabla_\theta F_{PG}] \approx \text{Var}[R(\mathbf{a})] \cdot \text{Var}[\nabla_\theta \log p(\mathbf{a};\theta)]$
  - Second term is a **sum of T uncorrelated terms** — grows linearly with episode length $T$
- **ES**: $\text{Var}[\nabla_\theta F_{ES}] \approx \text{Var}[R(\mathbf{a})] \cdot \text{Var}[\nabla_\theta \log p(\tilde{\theta};\theta)]$
  - Second term is **independent of T**

**Conclusion**: ES has lower gradient variance for long episodes with long-lasting action effects. PG methods compensate via reward discounting (which biases gradient estimates) and value function approximation.

### 3.2 Problem Dimensionality

ES gradient estimate = randomized finite differences in high-dimensional space:
$$\nabla_\theta \eta(\theta) = \mathbb{E}_{\epsilon \sim \mathcal{N}(0,I)} \{ (F(\theta + \sigma\epsilon) - F(\theta)) \epsilon/\sigma \}$$

Theoretical concern: required optimization steps scales with parameter dimension. In practice, the **intrinsic dimension** of the problem matters more than the model dimension. Larger networks often work better with ES (fewer local minima).

### 3.3 Advantages of Not Calculating Gradients

- ~2/3 less computation per episode (no backprop)
- No exploding gradient problems
- Supports non-differentiable architecture components (e.g., hard attention)
- Suited for low-precision hardware (binary networks, TPUs)
- Invariant to action frequency (frame-skip) — no need to tune this parameter

---

## 4. Experiments

### 4.1 MuJoCo

Environments: HalfCheetah, Hopper, InvertedDoublePendulum, InvertedPendulum, Swimmer, Walker2d.

Architecture: MLP with two 64-unit hidden layers, tanh activations. Actions discretized into 10 bins for hopping/swimming tasks.

**ES vs. TRPO** (ratio of ES timesteps / TRPO timesteps to reach same performance):

| Environment           | 25%  | 50%  | 75%  | 100% |
|-----------------------|------|------|------|------|
| HalfCheetah           | 0.15 | 0.49 | 0.42 | 0.58 |
| Hopper                | 0.53 | 3.64 | 6.05 | 6.94 |
| InvertedDoublePendulum| 0.46 | 0.48 | 0.49 | 1.23 |
| InvertedPendulum      | 0.28 | 0.52 | 0.78 | 0.88 |
| Swimmer               | 0.56 | 0.47 | 0.53 | 0.30 |
| Walker2d              | 0.41 | 5.69 | 8.02 | 7.88 |

Values < 1 mean ES is more sample-efficient than TRPO. ES solves most environments within 10x sample penalty on hard tasks.

### 4.2 Atari

- 51 Atari 2600 games from OpenAI Gym
- Same CNN architecture as Mnih et al. [2016]
- Trained for 1 billion frames (equivalent compute to A3C's 1-day results using 320M frames)
- ES better than A3C on **23 games**, worse on **28 games**

### 4.3 Parallelization

Test task: 3D Humanoid walking (MuJoCo)
- 1 machine (18 cores): ~657 minutes
- 80 machines (1,440 cores): ~10 minutes
- **Linear speedup** achieved in the number of CPU cores

### 4.4 Invariance to Temporal Resolution

ES gradient estimate is invariant to episode length — therefore invariant to frame-skip parameter. Demonstrated on Atari Pong with frame-skip $\in \{1, 2, 3, 4\}$: all converge in ~100 weight updates with similar learning curves.

---

## 5. Related Work

- Neuroevolution for RL: Risi & Togelius [2015], Hausknecht et al. [2014], Koutník et al. [2010, 2013]
- Natural evolution strategies: Wierstra et al. [2008, 2014]
- Recurrent network evolution: Schmidhuber et al. [2007]
- Hybrid black-box + policy gradient: Usunier et al. [2016]
- Hyper-NEAT: Stanley et al. [2009]

---

## 6. Conclusion

ES is a viable, competitive alternative to deep RL on hard benchmarks. Key advantages:
- Highly parallelizable (linear scaling)
- Invariant to action frequency and delayed rewards
- No temporal discounting or value function approximation needed

**Future work**: apply ES to problems where MDP-based RL is ill-suited — long horizons, complicated reward structures, meta-learning. Also: combine ES with low-precision neural network implementations.

---

## Implementation Notes for Replication

- Use **shared random seeds** between workers to avoid communicating perturbation vectors
- Apply **mirrored sampling** (always evaluate $+\epsilon$ and $-\epsilon$ pairs)
- Apply **rank normalization** (fitness shaping) to returns before gradient update
- Use **virtual batch normalization** for Atari pixel-based tasks
- **Discretize actions** for MuJoCo hopping/swimming tasks (10 bins)
- Fix episode length cap $m = 2 \times$ mean steps (dynamic)
- $\sigma$ is a fixed hyperparameter — do not adapt during training
- Apply **weight decay** to policy network
