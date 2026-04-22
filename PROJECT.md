# PROJECT.md — Start Trek: Lunar Lander (Reinforcement Learning)

> **Audience** — This file is the single source of truth for any contributor (human or AI, including Claude Code) working on this project. Read it fully before touching code. It encodes **hard constraints**, the **MVP scope**, the **acceptance criteria**, and a **scalable architecture** that allows bonus features to be layered on without rewriting the core.

---

## 1. Project Summary

Train a **reinforcement-learning agent** to autonomously land a lunar module in the **Gymnasium `LunarLander-v3`** simulator. The agent must learn from trial and error using **sparse, shaped rewards** (per-step penalties + strong terminal bonus on safe landing).

- **Domain**: Epitech — AIA-401 "Start Trek"
- **Paradigm**: Value-based deep RL (DQN family) on discrete action space, with a path open to continuous control and policy-gradient methods as bonus.
- **Success metric**: **mean episodic return ≥ 200** over **100 consecutive evaluation episodes** (without wind).

---

## 2. Hard Constraints (NON-NEGOTIABLE)

These come directly from the subject. **Do NOT violate them**, even for convenience.

| Constraint            | Value                                                                 |
| --------------------- | --------------------------------------------------------------------- |
| Language              | **Python** (3.10+ recommended)                                        |
| Binary name           | `*.py` (no compiled binary)                                           |
| Compilation           | **None**                                                              |
| Authorized RL/ML libs | **`gymnasium`**, **`pytorch`** (a.k.a. `torch`)                       |
| Additional documents  | Anything used as support during defense                               |

### 2.1. Library Policy

The subject explicitly authorizes **`gymnasium`** and **`pytorch`** only for the RL/ML core. The following utility libraries are considered **necessary plumbing** to produce the required deliverables (YAML configs, plots, videos) and are used minimally:

| Library            | Purpose                                                 | Justification                         |
| ------------------ | ------------------------------------------------------- | ------------------------------------- |
| `numpy`            | Tensor/array math                                       | Transitive dep of torch & gymnasium   |
| `pyyaml`           | Load/dump YAML run configs                              | Explicitly required by deliverables   |
| `matplotlib`       | Returns / loss / epsilon plots                          | Required for "tracking & assets"      |
| `imageio` / `imageio-ffmpeg` | Save MP4/GIF episode videos                   | Required for "videos" deliverable     |
| Python stdlib      | `argparse`, `dataclasses`, `pathlib`, `csv`, `json`, `logging`, `random`, `time` | No external dep |

> **Banned for the core RL code**: `stable-baselines3`, `ray[rllib]`, `cleanrl`, `tianshou`, `keras`, `tensorflow`, `jax`, any pre-built agent library. The learning algorithm must be **hand-written** on top of `torch` and `gymnasium`.

> **Also banned for the MVP**: any hyperparameter-tuning library (`optuna`, `hyperopt`, `wandb sweeps`). Manual + scripted grid search only.

> **If in doubt, ask the school before adding a dependency.** Keep `requirements.txt` short and justified.

---

## 3. Environment Specification

### 3.1. Env ID

```python
import gymnasium as gym
env = gym.make("LunarLander-v3", render_mode="rgb_array")  # discrete by default
```

### 3.2. State (8D continuous)

| Index | Name                  | Meaning                                 |
| ----- | --------------------- | --------------------------------------- |
| 0     | `x`                   | horizontal position                     |
| 1     | `y`                   | vertical position                       |
| 2     | `v_x`                 | horizontal velocity                     |
| 3     | `v_y`                 | vertical velocity                       |
| 4     | `theta`               | lander angle                            |
| 5     | `theta_dot`           | angular velocity                        |
| 6     | `left_leg_contact`    | 1 if left leg touches ground, else 0    |
| 7     | `right_leg_contact`   | 1 if right leg touches ground, else 0   |

### 3.3. Actions (discrete, MVP)

`Discrete(4)`:

- `0` — do nothing
- `1` — fire left thruster
- `2` — fire main engine
- `3` — fire right thruster

### 3.4. Reward Shape (read-only, defined by the env)

The environment already shapes reward to encourage:

- proximity to the landing pad
- low velocity
- horizontal orientation
- both legs landing
- per-step penalty when thrusters are firing
- ≈ **+100** for safe landing
- ≈ **−100** for crash

The agent **does not** modify the reward function; it may only clip it as an ablation.

### 3.5. Episode Termination

Gymnasium returns `terminated` and `truncated`. The project MUST distinguish and log the **reason**:

- **Crash** — body touches ground (not via legs) → `terminated = True`, large negative reward
- **Safe landing** — both legs on pad, velocity small → `terminated = True`, large positive reward
- **Out-of-view** — lander leaves the frame → `terminated = True`
- **Sleep / idle** — Box2D "asleep" physics, lander at rest → `terminated = True`
- **Truncated** — max-step limit reached without termination → `truncated = True`

A `utils/termination.py` module MUST classify each episode end into one of these categories from the final `info` dict + last reward + state.

---

## 4. Scalable Architecture

This is the **mandatory layout**. It separates concerns so bonus features drop in as new files, not as edits to existing code.

### 4.1. Repository Layout

```
lunar-lander-rl/
├── PROJECT.md                  # this file
├── README.md                   # how to run, results summary
├── requirements.txt
├── reproduce.sh                # one-click repro (see §7)
│
├── configs/                    # YAML per run
│   ├── baseline/
│   │   ├── random.yaml
│   │   ├── heuristic.yaml
│   │   └── dqn.yaml
│   ├── ablations/
│   │   ├── dqn_no_target.yaml
│   │   ├── dqn_small_buffer.yaml
│   │   └── dqn_obs_norm.yaml
│   └── bonus/                  # populated as bonuses are added
│
├── src/
│   ├── __init__.py
│   │
│   ├── agents/                 # every agent implements src/agents/base.py
│   │   ├── base.py             # abstract Agent (act, observe, learn, save, load)
│   │   ├── random_agent.py
│   │   ├── heuristic_agent.py
│   │   ├── dqn.py              # MVP agent
│   │   └── registry.py         # name -> class; add new agents here
│   │
│   ├── networks/               # torch.nn.Module factories
│   │   ├── mlp.py              # 2-3 ReLU layers, configurable widths
│   │   └── __init__.py
│   │
│   ├── memory/
│   │   ├── replay_buffer.py    # uniform replay (MVP)
│   │   └── __init__.py         # later: prioritized, n-step, etc.
│   │
│   ├── env/
│   │   ├── factory.py          # build_env(cfg) -> wrapped gym.Env
│   │   └── wrappers.py         # ObsNormalize, RewardClip, RecordEpisodeStats
│   │
│   ├── training/
│   │   ├── trainer.py          # the training loop (env-agnostic, agent-agnostic)
│   │   ├── evaluator.py        # deterministic eval for 100 episodes
│   │   └── schedules.py        # epsilon decay, learning-rate schedules
│   │
│   └── utils/
│       ├── seeding.py          # seeds python/numpy/torch/env
│       ├── config.py           # load + validate YAML; dataclass configs
│       ├── logger.py           # CSV + stdout; metric names stable across runs
│       ├── metrics.py          # running mean, 95% CI, etc.
│       ├── termination.py      # classify episode end reason
│       ├── plotting.py         # returns/loss/epsilon/episode_length plots
│       └── video.py            # save MP4 with imageio
│
├── train.py                    # thin CLI entrypoint → training.trainer
├── eval.py                     # thin CLI entrypoint → training.evaluator
│
├── runs/                       # ignored by git; one folder per run
│   └── <run_name>/<seed>/
│       ├── config.yaml         # exact config used
│       ├── metrics.csv         # per-episode metrics
│       ├── events.log          # stdout mirror
│       ├── checkpoints/*.pt
│       └── videos/*.mp4
│
├── assets/                     # final plots + videos committed to repo
│   ├── plots/
│   └── videos/
│
├── report/
│   └── report.pdf              # 4-6 page deliverable
│
└── tests/                      # minimal unit tests, §9
    ├── test_replay_buffer.py
    ├── test_termination.py
    ├── test_seeding.py
    └── test_env_factory.py
```

### 4.2. Core Interfaces (enforce these)

**`src/agents/base.py`** — every agent, including bonus ones, implements:

```python
class Agent(ABC):
    @abstractmethod
    def act(self, obs, *, greedy: bool = False) -> int | np.ndarray: ...
    @abstractmethod
    def observe(self, transition: Transition) -> None: ...     # push to buffer
    @abstractmethod
    def learn(self) -> dict[str, float]: ...                   # one grad step; returns loss/metrics
    @abstractmethod
    def save(self, path: Path) -> None: ...
    @abstractmethod
    def load(self, path: Path) -> None: ...
```

**`src/agents/registry.py`** — string name → class. Adding a new agent = add one line.

```python
AGENTS = {
    "random":    RandomAgent,
    "heuristic": HeuristicAgent,
    "dqn":       DQNAgent,
    # "double_dqn": DoubleDQNAgent,   # bonus
    # "ppo":        PPOAgent,          # bonus
}
```

**`src/env/factory.py`** — everything env-related is built from config:

```python
def build_env(cfg: EnvConfig, *, seed: int, render_mode: str | None = None) -> gym.Env:
    env = gym.make(cfg.id, render_mode=render_mode, **cfg.kwargs)
    for wrapper_cfg in cfg.wrappers:
        env = WRAPPERS[wrapper_cfg.name](env, **wrapper_cfg.kwargs)
    env.reset(seed=seed)
    return env
```

**`src/training/trainer.py`** — is **agent-agnostic and env-agnostic**. It only knows the `Agent` interface and the `gym.Env` protocol. Adding a new agent or env MUST NOT require editing the trainer.

### 4.3. Config-Driven Everything

Every run is fully described by one YAML. Example:

```yaml
# configs/baseline/dqn.yaml
run_name: dqn_baseline
seed: 0                        # single-seed run; repro.sh sweeps 0..4

env:
  id: LunarLander-v3
  kwargs: {}                   # empty = discrete, no wind
  wrappers:
    - name: RecordEpisodeStats
      kwargs: {}

agent:
  name: dqn
  network:
    hidden_sizes: [128, 128]
    activation: relu
  gamma: 0.99
  lr: 5.0e-4
  batch_size: 64
  buffer_size: 100000
  min_buffer_size: 1000
  target_update: { kind: hard, every_steps: 1000 }
  epsilon: { start: 1.0, end: 0.05, decay_steps: 100000 }
  grad_clip: 10.0

training:
  total_steps: 500000
  log_every: 1000
  eval_every: 10000
  eval_episodes: 20
  save_every: 50000

eval:
  episodes: 100
  greedy: true
  record_video: true
  video_every: 20
```

No magic constants in code. If you're tempted to hardcode, add a config key.

---

## 5. Mandatory Deliverables

Reproduced verbatim from the subject, with file-level expectations.

| Deliverable              | Concrete output                                                                 |
| ------------------------ | ------------------------------------------------------------------------------- |
| Codebase                 | Clean repo; `train.py`, `eval.py`, `README.md`; layout per §4.1                 |
| YAML configs             | One YAML per run under `configs/` (env, algo, hypers, seeds)                    |
| Tracking plots           | Returns per episode, episode length, epsilon, loss → `assets/plots/*.png`       |
| Termination-cause stats  | Histogram / table of crash / land / out-of-view / sleep / truncated ratios      |
| Videos                   | ≥ 1 video of random policy, ≥ 1 of heuristic, ≥ 1 of trained agent → `assets/videos/` |
| Report (4–6 pages)       | `report/report.pdf` — setup, results, ≥1 ablation, limits, next steps           |
| One-click repro          | `reproduce.sh` regenerates all figures + CSVs from seeds 0..4                   |

---

## 6. Required Tasks (MVP)

Execute and deliver **all five** before any bonus work.

### 6.1. Baseline & Instrumentation

- Implement `RandomAgent` and a hand-coded **heuristic** agent (e.g., PD-style rules on `theta`, `v_y`, `x`).
- Log per episode: `return`, `length`, `termination_reason`.
- Save at least one episode video for each.

### 6.2. Agent (Discrete) — DQN

- **Network**: MLP, 2–3 hidden layers with ReLU, input dim 8, output dim 4 (Q-values).
- **Replay buffer**: uniform, capacity ≥ 1e5.
- **Target network**: periodic hard copy (every N steps) — soft/Polyak is an ablation option.
- **Exploration**: ε-greedy with linear decay from ε₀=1.0 to ε_min=0.05.
- **Optimizer**: Adam; MSE or Huber loss; gradient clipping.
- **Plots**: return, ε, loss over training steps.

### 6.3. Ablation (≥ 1 required, prefer ≥ 2)

Pick at least one, prefer two. Run with identical seeds against the baseline.

- Buffer size (1e4 vs 1e5 vs 1e6)
- Target update: **hard** (every N steps) vs **soft** (Polyak τ)
- Observation normalization ON / OFF
- Reward clipping ON / OFF
- Network depth (2 vs 3 hidden layers)

For each ablation, produce a paired plot and a one-paragraph analysis in the report.

### 6.4. Reproducibility

- Run every reported configuration with **seeds 0, 1, 2, 3, 4** (minimum 5).
- For each key metric (mean return, solve step, final ε), report **mean ± 95 % CI**.
- `reproduce.sh` regenerates everything from these seeds alone.

### 6.5. Defense

- Be ready to justify every hyperparameter, architectural choice, and ablation result with structural + functional arguments.
- Prepare slides or equivalent support media (free format per subject).

---

## 7. Acceptance Criteria

A submission is acceptable **iff all three** hold:

1. **Performance** — mean score **≥ 200** over **100 consecutive evaluation episodes** (greedy, no wind), reported with distribution + variance.
2. **Logging** — `terminated` vs `truncated` and the termination **reason** (crash / land / out-of-view / sleep) are correctly detected and logged per episode.
3. **Reproducibility** — `bash reproduce.sh` regenerates every plot + CSV within **±5 %** of the committed figures, using the committed seeds and configs.

---

## 8. Logging & Metrics Contract

Metric column names in `metrics.csv` are **stable across runs** so plots can diff them:

```
episode, global_step, return, length, epsilon, loss_mean, loss_std,
termination_reason, terminated, truncated, wall_time_s
```

Aggregated per seed → `runs/<run>/summary.json`:

```json
{
  "mean_return_last_100": 231.4,
  "std_return_last_100": 18.2,
  "ci95_return_last_100": [227.8, 235.0],
  "solved": true,
  "first_solve_step": 283000,
  "termination_breakdown": {"land": 0.94, "crash": 0.04, "out_of_view": 0.01, "truncated": 0.01}
}
```

---

## 9. Testing (lightweight but non-negotiable)

A small test suite under `tests/` guards the non-RL code:

- `test_replay_buffer.py` — push/sample shapes, no NaNs, FIFO eviction.
- `test_termination.py` — synthetic `info` dicts map to expected reason.
- `test_seeding.py` — two runs with same seed produce identical first 100 transitions.
- `test_env_factory.py` — config → env yields correct obs shape, action space.

Run with `python -m unittest discover tests/` (stdlib, no extra dep).

---

## 10. Reproducibility Rules

- Seed `random`, `numpy`, `torch` (CPU + CUDA), and `env.reset(seed=...)` via `utils/seeding.py`.
- `torch.use_deterministic_algorithms(True)` where feasible; otherwise document the non-determinism source.
- Persist the **exact config** used into each run folder; never read from a mutable global.
- Pin dependency versions in `requirements.txt`.

---

## 11. Commands (what `claude code` / a human types)

```bash
# Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train one run
python train.py --config configs/baseline/dqn.yaml --seed 0

# Evaluate a checkpoint on 100 episodes (greedy)
python eval.py --run runs/dqn_baseline/0 --episodes 100 --greedy

# Reproduce everything
bash reproduce.sh

# Tests
python -m unittest discover tests/
```

---

## 12. Bonus / Extension Roadmap (NOT part of MVP)

The architecture in §4 is built so these land as **additive** changes. Each bonus is a new config + a new file (or handful of files) in its own module. **No edits to the trainer, env factory, or base interfaces should be required.**

### 12.1. Algorithmic bonuses

| Bonus                          | New files                                             | Impact                                |
| ------------------------------ | ----------------------------------------------------- | ------------------------------------- |
| Double DQN                     | `src/agents/double_dqn.py` + registry entry          | reduces maximization bias             |
| Dueling DQN                    | `src/networks/dueling_mlp.py` + agent config key     | better state-value estimation         |
| Prioritized Experience Replay  | `src/memory/prioritized_buffer.py`                   | faster convergence                    |
| N-step returns                 | `src/memory/n_step_wrapper.py`                        | lower bias, higher variance           |
| Noisy Nets                     | `src/networks/noisy_layers.py`                        | replace ε-greedy                      |
| Rainbow (combo of the above)   | new agent class                                       | strongest DQN variant                 |
| PPO (on-policy)                | `src/agents/ppo.py` + `src/memory/rollout_buffer.py` | policy gradient baseline              |
| SAC / DDPG (continuous)        | `src/agents/sac.py` etc.                              | needs `continuous=True` env           |

### 12.2. Environment bonuses

- **Wind** — `gym.make("LunarLander-v3", enable_wind=True, wind_power=15.0, turbulence_power=1.5)`. Subject: "solved without wind; wind is a bonus."
- **Continuous control** — `continuous=True`. Requires a continuous-action agent (SAC/DDPG/PPO-continuous).

### 12.3. Engineering bonuses

- TensorBoard or CSV-viewer dashboard
- Multi-seed **parallel** training driver (`multiprocessing`, no new lib)
- Checkpoint → ONNX export (torch only)
- Animated rollout comparing random / heuristic / trained side-by-side
- GitHub Actions CI running the test suite

### 12.4. Rules for adding a bonus

1. Open a new YAML under `configs/bonus/`.
2. Add the new agent / wrapper / buffer as a new file under the matching module.
3. Register it in the relevant registry (agents/wrappers).
4. **Do not modify** `trainer.py`, `evaluator.py`, or `agents/base.py` unless the bonus genuinely exposes a missing abstraction — in which case update this `PROJECT.md` in the same PR.
5. Ship with at least one seed run and a short "Bonus X" section appended to the report.

---

## 13. Coding Standards

- **Python 3.10+**, type hints everywhere public.
- **`dataclasses`** for configs, not dicts, past the YAML boundary.
- **No global state** except the seeded RNGs; everything else is passed in.
- **Docstrings** on every public class/function, short but explicit on shapes and units.
- **Format**: `black`-compatible style, 100-col lines. (Formatting is local-only; no CI requirement.)
- **Logging**: use `utils/logger.py`, not bare `print`, except in CLI entrypoints.
- **No dead code, no commented-out blocks** committed.

---

## 14. Defense Checklist

Going into the defense, be able to answer:

- Why DQN and not just tabular Q-learning? (continuous 8D state)
- Why a replay buffer? Why a target network? (stability, decorrelation)
- What does ε-greedy trade off, and why decay it?
- What does each ablation tell you **quantitatively**?
- Where does your agent still fail (terminations breakdown) and why?
- What would you try next if you had another week? (sparse-reward tricks, noisy nets, PPO…)
- How would your code support training on a different Gymnasium env **without edits to the trainer**?

---

## 15. Out of Scope

- Any environment other than `LunarLander-v3` (unless as a quick sanity check).
- Any algorithm that is not a value-based or policy-gradient method implemented from scratch on torch.
- Any library outside §2.1.
- Beating SOTA — the goal is a **correct, reproducible, well-understood** solution at or above the 200 threshold.

---

## 16. Version

`PROJECT.md v1.0.0` — aligned with subject `v0.0.1`.