# Start Trek — Lunar Lander (Reinforcement Learning)

> Projet Epitech **AIA-401** — entraînement d'un agent RL qui apprend à poser un module lunaire dans l'environnement [`gymnasium` `LunarLander-v3`](https://gymnasium.farama.org/environments/box2d/lunar_lander/).

Le contrat complet (contraintes dures, critères d'acceptation, livrables) est dans [`PROJECT.md`](./PROJECT.md). La feuille de route d'exécution est dans [`PLAN.md`](./PLAN.md). L'historique des décisions est dans [`DOCUMENTATION.md`](./DOCUMENTATION.md).

---

## Objectif

Atteindre **mean episodic return ≥ 200** sur **100 épisodes consécutifs** d'évaluation greedy (sans vent), via un **DQN hand-written** sur `torch` + `gymnasium` uniquement.

---

## Installation

Python 3.10+ requis.

```bash
python -m venv .venv
source .venv/bin/activate        # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

> `gymnasium[box2d]` tire `box2d-py` qui a besoin de `swig` installé sur le système (Ubuntu : `sudo apt install swig`, macOS : `brew install swig`).

---

## Commandes principales

_Les scripts ci-dessous seront introduits progressivement au cours des tickets de Phase 0 (cf. `PLAN.md`)._

```bash
# Entraîner une config (un seed)
python train.py --config configs/baseline/dqn.yaml --seed 0

# Évaluer un checkpoint sur 100 épisodes greedy
python eval.py --run runs/dqn_baseline/0 --episodes 100 --greedy

# Reproduire tous les résultats committés (15 runs = 3 configs × 5 seeds)
bash reproduce.sh

# Lancer la suite de tests
python -m unittest discover tests/
```

### Reproduire les résultats (`reproduce.sh`, T3.5)

`bash reproduce.sh` régénère **tout** depuis les configs et seeds committés :

- **15 trainings** = 3 configs (`baseline/dqn`, `ablations/dqn_buffer10k`, `ablations/dqn_depth3`) × seeds `0..4` ;
- les 3 `runs/<run>/summary_aggregated.json` (agrégation + IC95, via `scripts/aggregate_runs.py`) ;
- les **5 plots single-run** (`scripts/make_plots.py`) + les **5 plots agrégés/paired** (`scripts/aggregated_plots.py`) dans `assets/plots/`.

Chaque run est entièrement seedé : les chiffres régénérés sont à **±5 %** des figures committées (critère §7.3 de `PROJECT.md`). Les 5 seeds d'une config tournent en parallèle (runs indépendants et seedés → mêmes résultats, seul le wall-time change) ; les configs s'enchaînent.

Variables d'environnement : `SEEDS="0 1 2 3 4"`, `PY=...` (python à utiliser), `SKIP_EXISTING=1` (sauter un `(run, seed)` déjà calculé), `JOBS_THREADS=3` (threads torch/OMP par worker parallèle).

```bash
bash reproduce.sh                       # tout, seeds 0..4
SEEDS="0 1" SKIP_EXISTING=1 bash reproduce.sh   # partiel / reprise
```

---

## Structure du dépôt

```
.
├── PROJECT.md              # Hub : sujet, contraintes dures, MVP, livrables
├── PLAN.md                 # Feuille de route (tickets, phases, critères)
├── DOCUMENTATION.md        # Journal technique (changelog, décisions)
├── README.md               # Ce fichier
├── requirements.txt
├── reproduce.sh            # (Phase 3) régénère tous les plots et CSV
│
├── configs/                # Une YAML par run
│   ├── baseline/
│   ├── ablations/
│   └── bonus/
│
├── src/
│   ├── agents/             # Random, Heuristic, DQN, ... (registre)
│   ├── networks/           # MLP (+ bonus : Dueling, Noisy)
│   ├── memory/             # ReplayBuffer (+ bonus : PER, N-step)
│   ├── env/                # factory + wrappers
│   ├── training/           # trainer, evaluator, schedules
│   └── utils/              # seeding, config, logger, metrics,
│                           #   termination, plotting, video
│
├── train.py                # (Phase 0) CLI : --config, --seed, --out-dir
├── eval.py                 # (Phase 0) CLI : --run, --episodes, --greedy
│
├── runs/                   # artefacts de training (gitignored)
├── assets/                 # plots + vidéos committés pour le rapport
│   ├── plots/
│   └── videos/
├── report/                 # rapport PDF final (4-6 pages)
└── tests/                  # unittest — stdlib uniquement
```

---

## Statut du projet

Voir `PLAN.md` pour l'état exact des tickets. À date :

- [x] Documents fondateurs rédigés (`PROJECT.md`, `PLAN.md`, `DOCUMENTATION.md`).
- [x] Arborescence + `.gitignore` + `requirements.txt` + `README.md` squelette (T0.1).
- [ ] Phase 0 en cours — infra sans logique RL.

---

## Crédits

Projet réalisé par Gabin Wilson (Epitech AIA-401 "Start Trek", promo 2026).
