# Start Trek — Lunar Lander (Reinforcement Learning)

> Projet Epitech **AIA-401**. Un agent **DQN écrit à la main** (PyTorch + Gymnasium uniquement) qui apprend à poser un module lunaire dans l'environnement [`LunarLander-v3`](https://gymnasium.farama.org/environments/box2d/lunar_lander/).

Ce README est autosuffisant (installation, commandes, structure, résultats). Le rapport scientifique détaillé est dans `report/report.md`.

---

## Table des matières

1. [Qu'est-ce que c'est](#1-quest-ce-que-cest)
2. [Installation](#2-installation)
3. [Démarrage rapide](#3-démarrage-rapide)
4. [Lancer un entraînement (`train.py`)](#4-lancer-un-entraînement-trainpy)
5. [Évaluer un agent (`eval.py`)](#5-évaluer-un-agent-evalpy)
6. [Configs disponibles](#6-configs-disponibles)
7. [Anatomie d'une config YAML](#7-anatomie-dune-config-yaml)
8. [Reproduire tous les résultats (`reproduce.sh`)](#8-reproduire-tous-les-résultats-reproducesh)
9. [Plots & vidéos](#9-plots--vidéos)
10. [Sorties d'un run](#10-sorties-dun-run)
11. [Tests](#11-tests)
12. [Structure du dépôt](#12-structure-du-dépôt)
13. [Résultats principaux](#13-résultats-principaux)
14. [Dépannage](#14-dépannage)
15. [Crédits](#15-crédits)

---

## 1. Qu'est-ce que c'est

LunarLander : un vaisseau doit atterrir entre deux drapeaux. Observation = vecteur de **8 floats** (position, vitesse, angle, vitesse angulaire, contacts des 2 jambes) ; **4 actions discrètes** (rien, moteur gauche, moteur principal, moteur droit).

**Objectif (critère d'acceptation du sujet)** : atteindre un **mean episodic return ≥ 200** sur **100 épisodes consécutifs** en évaluation *greedy* (sans vent), reporté avec distribution et IC95, sur les seeds `0..4`.

Contraintes : **Python uniquement**, aucune lib RL/ML hors `gymnasium` + `torch` (+ utilitaires de plomberie justifiés : `numpy`, `pyyaml`, `matplotlib`, `imageio`). Aucun outil de tuning automatique.

Agents fournis (registre `src/agents/registry.py`) :

| Nom | Rôle |
|-----|------|
| `random` | Politique aléatoire uniforme (baseline plancher) |
| `heuristic` | Contrôleur PD codé à la main (baseline forte) |
| `dqn` | **Deep Q-Network** — l'agent principal du projet |
| `noop` | Agent inerte (smoke-test de la chaîne) |

---

## 2. Installation

**Prérequis : Python 3.10+** et `swig` (nécessaire à `box2d-py`, tiré par `gymnasium[box2d]`) :

- Ubuntu/Debian : `sudo apt install swig`
- macOS : `brew install swig`
- Windows : utiliser WSL (recommandé) ou installer `swig` via `choco`/`scoop`.

### Option A — `make` (recommandé)

```bash
make install        # crée .venv/ et installe requirements.txt
```

### Option B — manuel

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Dépendances **épinglées** (`requirements.txt`) : `gymnasium[box2d]==1.0.0`, `torch==2.5.1`, `numpy==2.1.3`, `pyyaml==6.0.2`, `matplotlib==3.9.2`, `imageio==2.36.1`, `imageio-ffmpeg==0.5.1`.

Vérifier l'installation :

```bash
.venv/bin/python -c "import gymnasium, torch, yaml, matplotlib, imageio; import gymnasium as gym; gym.make('LunarLander-v3'); print('OK')"
```

> Les commandes ci-dessous utilisent `.venv/bin/python`. Si la venv est activée (`source .venv/bin/activate`), `python` suffit. `make` utilise toujours `.venv/bin/python` automatiquement.

---

## 3. Démarrage rapide

```bash
# 1. Smoke-test (agent inerte, ~quelques secondes) — valide toute la chaîne
.venv/bin/python train.py --config configs/smoke.yaml --seed 0

# 2. Entraîner le DQN de référence (1 seed, ~30–50 min CPU)
.venv/bin/python train.py --config configs/baseline/dqn.yaml --seed 0

# 3. Évaluer le meilleur checkpoint sur 100 épisodes greedy
.venv/bin/python eval.py --run runs/dqn_baseline/0 --episodes 100 --greedy \
  --checkpoint runs/dqn_baseline/0/checkpoints/best.pt

# 4. Lancer la suite de tests
.venv/bin/python -m unittest discover tests/
```

---

## 4. Lancer un entraînement (`train.py`)

```bash
.venv/bin/python train.py --config <chemin.yaml> [--seed <int>] [--out-dir <dir>]
```

| Argument | Défaut | Rôle |
|----------|--------|------|
| `--config` | *(requis)* | Chemin vers la config YAML |
| `--seed` | valeur du YAML | Surcharge le seed (Python/NumPy/Torch/env tous seedés) |
| `--out-dir` | `runs` | Racine des artefacts |

Le run est écrit dans **`<out-dir>/<run_name>/<seed>/`** (où `run_name` vient du YAML). Tout est piloté par la config : `train.py` est un thin CLI, aucune logique spécifique à un agent.

Exemples :

```bash
make train CONFIG=configs/baseline/dqn.yaml SEED=0      # via Makefile
.venv/bin/python train.py --config configs/baseline/heuristic.yaml --seed 3
.venv/bin/python train.py --config configs/ablations/dqn_depth3.yaml --seed 2
```

---

## 5. Évaluer un agent (`eval.py`)

```bash
.venv/bin/python eval.py --run <run_dir> [options]
```

| Argument | Défaut | Rôle |
|----------|--------|------|
| `--run` | *(requis)* | Dossier du run (contient `config.yaml`) |
| `--episodes` | `eval.episodes` du YAML | Nombre d'épisodes d'évaluation |
| `--greedy` / `--no-greedy` | valeur du YAML | Force la politique greedy (argmax Q) |
| `--record-video` / `--no-record-video` | valeur du YAML | Enregistre une vidéo MP4 |
| `--checkpoint` | `<run>/checkpoints/final.pt` | Checkpoint `.pt` à charger |
| `--seed` | `cfg.seed` | Seed pour le reset de l'env |

Produit `runs/<run>/<seed>/eval_<timestamp>.json` : `mean_return`, `std_return`, `ci95`, `termination_breakdown`, `solved`.

> ⚠️ **Important — `best.pt` vs `final.pt`.** Le DQN vanilla s'effondre en fin d'entraînement (instabilité tardive bien documentée). Le trainer garde donc un **best-checkpoint** (`checkpoints/best.pt`) qui capture le pic ; c'est lui qui fournit le chiffre « ≥ 200 » reporté dans `summary.json`. Or `eval.py` charge **`final.pt` par défaut** (qui, lui, est effondré). Pour reproduire les chiffres du rapport, **passez explicitement** `--checkpoint <run>/checkpoints/best.pt`.

```bash
make eval RUN=runs/dqn_baseline/0                        # final.pt (effondré, attendu)
.venv/bin/python eval.py --run runs/dqn_baseline/0 --episodes 100 --greedy \
  --checkpoint runs/dqn_baseline/0/checkpoints/best.pt    # le chiffre ≥ 200
```

---

## 6. Configs disponibles

| Config | Agent | Description |
|--------|-------|-------------|
| `configs/smoke.yaml` | `noop` | Smoke-test rapide de la chaîne complète |
| `configs/baseline/random.yaml` | `random` | Baseline plancher (~ −150 de return) |
| `configs/baseline/heuristic.yaml` | `heuristic` | Baseline forte (contrôleur PD) |
| `configs/baseline/dqn.yaml` | `dqn` | **DQN de référence** (objectif ≥ 200) |
| `configs/ablations/dqn_buffer10k.yaml` | `dqn` | **Ablation A** : `buffer_size` 100k → 10k |
| `configs/ablations/dqn_depth3.yaml` | `dqn` | **Ablation B** : réseau `[128,128]` → `[128,128,128]` |

Chaque ablation ne change **qu'un seul paramètre** par rapport à `configs/baseline/dqn.yaml`.

---

## 7. Anatomie d'une config YAML

Extrait commenté de `configs/baseline/dqn.yaml` :

```yaml
run_name: dqn_baseline      # dossier de sortie : runs/<run_name>/<seed>/
seed: 0                     # seed par défaut (surchargé par --seed)

env:
  id: LunarLander-v3
  kwargs: {}                # ex. {enable_wind: true} pour le bonus vent
  wrappers:                 # composés dans l'ordre listé
    - name: RecordEpisodeStats
      kwargs: {}
    # wrappers dispo : ObsNormalize, RewardClip

agent:
  name: dqn                 # clé du registre AGENTS
  network:
    hidden_sizes: [128, 128]
    activation: relu
  gamma: 0.99
  lr: 5.0e-4
  batch_size: 64
  buffer_size: 100000
  min_buffer_size: 1000     # pas d'apprentissage avant ce remplissage
  target_update:
    kind: hard              # copie dure du target net
    every_steps: 1000
  epsilon:                  # ε-greedy linéaire 1.0 → 0.05
    start: 1.0
    end: 0.05
    decay_steps: 100000
  grad_clip: 10.0

training:
  total_steps: 500000
  log_every: 1000
  eval_every: 10000
  eval_episodes: 20
  save_every: 50000         # checkpoint périodique

eval:
  episodes: 100             # 100 épisodes = le critère d'acceptation
  greedy: true
  record_video: true
  video_every: 20
```

Une clé absente ou inconnue lève une `ValueError` explicite (chemin YAML + clé). La config exacte de chaque run est re-sérialisée dans `runs/<run>/<seed>/config.yaml` (rejouable même si le YAML source change).

---

## 8. Reproduire tous les résultats (`reproduce.sh`)

`bash reproduce.sh` (ou `make reproduce`) régénère **tout** depuis les configs + seeds committés :

- **15 trainings** = 3 configs (`baseline/dqn`, `ablations/dqn_buffer10k`, `ablations/dqn_depth3`) × seeds `0..4` ;
- les 3 `runs/<run>/summary_aggregated.json` (agrégation moyenne + IC95, via `scripts/aggregate_runs.py`) ;
- les **5 plots single-run** (`scripts/make_plots.py`) + les **5 plots agrégés / paired** (`scripts/aggregated_plots.py`) dans `assets/plots/`.

Chaque run est entièrement seedé : les chiffres régénérés sont à **±5 %** des figures committées (tolérance de reproductibilité). Les 5 seeds d'une config tournent **en parallèle** (runs indépendants et seedés → mêmes résultats, seul le wall-time change) ; les 3 configs s'enchaînent.

Variables d'environnement :

| Variable | Défaut | Rôle |
|----------|--------|------|
| `SEEDS` | `"0 1 2 3 4"` | Seeds à lancer par config |
| `PY` | `.venv/bin/python` sinon `python3` | Interpréteur Python |
| `SKIP_EXISTING` | `0` | `1` = sauter un `(run, seed)` dont `summary.json` existe déjà |
| `JOBS_THREADS` | `3` | Threads torch/OMP par worker parallèle (évite l'oversubscription) |

```bash
bash reproduce.sh                                  # tout, seeds 0..4
SEEDS="0 1" SKIP_EXISTING=1 bash reproduce.sh      # partiel / reprise
SEEDS="0" bash reproduce.sh                        # smoke de la chaîne repro
```

> Coût indicatif : ~30–50 min CPU par run. Avec les 5 seeds d'une config en parallèle, comptez ~30–50 min par config, soit ~1h30–2h30 au total sur un laptop moderne (20 cœurs). Réduisez via `SEEDS`.

Régénérer **uniquement les plots** (si les runs existent déjà) :

```bash
.venv/bin/python scripts/make_plots.py --run runs/dqn_baseline/0 --out-dir assets/plots
.venv/bin/python scripts/aggregated_plots.py --seeds 0 1 2 3 4
```

---

## 9. Plots & vidéos

Tout est committé sous `assets/` (utilisé tel quel par le rapport).

**`assets/plots/` — single-run** (depuis `runs/dqn_baseline/0/metrics.csv`) :
`returns_per_episode.png`, `episode_length.png`, `epsilon.png`, `loss.png`, `termination_breakdown.png`.

**`assets/plots/` — agrégés multi-seed (moyenne + bande IC95)** :
`agg_returns_baseline.png`, `agg_returns_buffer10k.png`, `agg_returns_depth3.png`.

**`assets/plots/` — paired (ablation vs baseline)** :
`paired_buffer10k_vs_baseline.png`, `paired_depth3_vs_baseline.png`.

**`assets/videos/`** : `random.mp4`, `heuristic.mp4`, `dqn_trained.mp4` (reproductibles via `eval.py --record-video`).

Scripts associés : `scripts/aggregate_runs.py` (agrégation + IC95), `scripts/aggregated_plots.py` (les 3+2 plots), `scripts/paired_plot.py` (thin CLI paired), `scripts/validate_termination.py` (0 % de terminaison `UNKNOWN` sur trajectoires réelles).

---

## 10. Sorties d'un run

```
runs/<run_name>/
├── summary_aggregated.json     # agrégat multi-seed (mean, IC95, n_solved)
├── <seed>/
│   ├── config.yaml             # config exacte re-sérialisée (rejouable)
│   ├── metrics.csv             # 1 ligne / épisode (colonnes stables)
│   ├── events.log              # mirror lisible de stdout
│   ├── summary.json            # mean_return_last_100, ci95, first_solve_step…
│   └── checkpoints/
│       ├── best.pt             # meilleur checkpoint (→ chiffre reporté)
│       ├── final.pt            # dernier état (DQN vanilla : effondré)
│       └── step_*.pt           # checkpoints périodiques
```

Colonnes stables de `metrics.csv` :
`episode, global_step, return, length, epsilon, loss_mean, loss_std, termination_reason, terminated, truncated, wall_time_s`.

`termination_reason` ∈ `land | crash | out_of_view | sleep | truncated | unknown` (classifié par `src/utils/termination.py` ; `unknown` = 0 % sur trajectoires réelles, asserté en T1.3).

> `runs/` est **gitignoré**. Les artefacts légers (summaries, `metrics.csv`) sont committés à la main selon la politique du projet ; les `.pt` volumineux ne le sont pas.

---

## 11. Tests

```bash
make test
# ou
.venv/bin/python -m unittest discover tests/
```

**245 tests** (stdlib `unittest` uniquement, 1 skip sans rapport). Couverture : `replay_buffer`, `termination`, `seeding`, `env_factory` (les 4 tests obligatoires du sujet) + config, agents, schedules, logger, metrics, plotting, evaluator, vidéo.

---

## 12. Structure du dépôt

```
.
├── README.md                 # ce fichier (autosuffisant)
├── requirements.txt          # deps épinglées
├── Makefile                  # install / train / eval / test / reproduce / clean
├── reproduce.sh              # régénère 15 runs + summaries + 10 plots
│
├── train.py                  # CLI fin : --config --seed --out-dir
├── eval.py                   # CLI fin : --run --episodes --greedy --checkpoint
│
├── configs/
│   ├── baseline/             # random.yaml, heuristic.yaml, dqn.yaml
│   ├── ablations/            # dqn_buffer10k.yaml, dqn_depth3.yaml
│   ├── bonus/                # (Phase 5)
│   └── smoke.yaml
│
├── src/
│   ├── agents/               # base (ABC) + registry + random/heuristic/dqn/noop
│   ├── networks/             # build_mlp
│   ├── memory/               # ReplayBuffer (ring-buffer NumPy)
│   ├── env/                  # factory + wrappers (ObsNormalize, RewardClip…)
│   ├── training/             # trainer, evaluator, schedules (ε linéaire)
│   └── utils/                # seeding, config, logger, metrics,
│                             #   termination, plotting, video
│
├── scripts/                  # aggregate_runs, aggregated_plots, paired_plot,
│                             #   make_plots, validate_termination
├── tests/                    # unittest (245 tests)
├── runs/                     # artefacts de training (gitignoré)
├── assets/                   # plots + vidéos committés (rapport)
├── report/                   # report.md (rapport scientifique)
└── report/                   # rapport (brouillon → PDF final)
```

Ajouter un agent = **un fichier** dans `src/agents/` + une entrée registre + une config YAML ; aucun autre fichier à toucher.

---

## 13. Résultats principaux

Baseline et 2 ablations, **agrégés sur seeds 0..4**, `mean_return_last_100` (100 ép. greedy, best-checkpoint) :

| Run | mean return, IC95 | solved | `first_solve_step` |
|-----|-------------------|--------|--------------------|
| **Baseline** (2×128, buffer 100k) | **267.21** [256.19, 278.22] | **5/5** | ~132 000 |
| Ablation A — buffer 10k | 242.40 [236.08, 248.72] | 5/5 | ~94 000 |
| Ablation B — réseau 3×128 | 274.30 [265.25, 283.36] | 5/5 | ~122 000 |

- **Objectif ≥ 200 atteint** : 5/5 seeds sur la baseline (et sur les deux ablations).
- **Ablation A (buffer ÷10)** : dégradation **statistiquement significative** (IC95 disjoints, ≈ −25 pts) — la taille du buffer compte.
- **Ablation B (profondeur +1 couche)** : gain **non significatif** (IC95 recouvrants, +2.7 %) — le MLP 2×128 sature déjà la tâche.

Détails et figures dans `report/report.md` et `assets/plots/`.

---

## 14. Dépannage

| Symptôme | Cause / solution |
|----------|------------------|
| `error: command 'swig' failed` à l'install | Installer `swig` (cf. §2) puis réinstaller `gymnasium[box2d]`. |
| `python: command not found` | Utiliser `.venv/bin/python` ou activer la venv (`source .venv/bin/activate`). |
| `eval.py` donne un return très négatif | Comportement attendu avec `final.pt` (DQN vanilla effondré) ; passer `--checkpoint <run>/checkpoints/best.pt`. |
| `agent '<x>' not registered` | Vérifier `agent.name` dans le YAML vs `src/agents/registry.py` (`random`, `heuristic`, `dqn`, `noop`). |
| `reproduce.sh` très long | Réduire `SEEDS` (ex. `SEEDS="0"`), ou `SKIP_EXISTING=1` pour reprendre. |
| Machine qui rame pendant la repro | Baisser `JOBS_THREADS` ou le nombre de seeds parallèles via `SEEDS`. |

---

## 15. Crédits

Projet réalisé par **Gabin Wilson**, **Léo Batillot" — Epitech AIA-401 « Start Trek », promo 2026.
Environnement [`Gymnasium LunarLander-v3`](https://gymnasium.farama.org/environments/box2d/lunar_lander/) (Farama Foundation).
