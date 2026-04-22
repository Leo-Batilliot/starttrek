# PLAN.md — Feuille de route technique du projet Start Trek (Lunar Lander RL)

> Ce fichier est le **plan d'exécution détaillé** du projet. Il liste toutes les étapes prévues avec leurs attentes précises et le résultat final attendu à chaque jalon.
>
> - `PROJECT.md` = hub descriptif (sujet officiel, contraintes dures, MVP, livrables, bonus).
> - `PLAN.md` (ce fichier) = **quoi faire**, **dans quel ordre**, **avec quels critères d'acceptation**.
> - `DOCUMENTATION.md` = **historique des versions** (ce qui a été fait, quand, pourquoi, comment).
>
> **Règle** : chaque ticket terminé doit cocher sa case ici, et déclencher une entrée correspondante dans `DOCUMENTATION.md`.

---

## 🎯 Résultat final attendu (définition du "done")

À la fin du projet, le dépôt doit livrer :

1. Deux entrypoints CLI Python à la racine : `train.py` et `eval.py`, chacun piloté par une config YAML.
2. Une arborescence conforme à la section §4.1 de `PROJECT.md` (`src/agents`, `src/env`, `src/networks`, `src/memory`, `src/training`, `src/utils`, `configs/`, `runs/`, `assets/`, `tests/`, `report/`).
3. Conformité totale au sujet : langage Python uniquement, aucune lib RL/ML hors `gymnasium` + `torch` + utilitaires justifiés (§2.1 de `PROJECT.md`), aucun code de tuning automatique.
4. **Acceptance criteria §7 de `PROJECT.md`** :
   - **Performance** — mean episodic return **≥ 200** sur **100 épisodes consécutifs greedy, sans vent**, reporté avec distribution + variance.
   - **Logging** — `terminated` vs `truncated` + raison (crash / land / out_of_view / sleep) correctement détectés et loggés par épisode.
   - **Reproductibilité** — `bash reproduce.sh` régénère chaque plot et CSV à **±5 %** des figures commitées, à partir des seeds `0..4` commitées.
5. Jeu complet de configs YAML sous `configs/` :
   - `configs/baseline/` : `random.yaml`, `heuristic.yaml`, `dqn.yaml`.
   - `configs/ablations/` : au moins **2 ablations** exploitables (buffer size, target hard vs soft, obs norm, reward clip, depth réseau).
   - `configs/bonus/` : populé au fur et à mesure des bonus.
6. Livrables d'assets sous `assets/` :
   - Plots : `returns_per_episode.png`, `episode_length.png`, `epsilon.png`, `loss.png`, `termination_breakdown.png`, un paired-plot par ablation.
   - Vidéos : ≥ 1 random, ≥ 1 heuristic, ≥ 1 trained (greedy), par policy et par phase-clé du projet.
7. Rapport `report/report.pdf` de 4 à 6 pages (setup, résultats, ≥ 1 ablation, limites, next steps).
8. Script `reproduce.sh` à la racine qui régénère tout depuis les seeds `0..4` sans intervention manuelle.
9. Tests unitaires passant via `python -m unittest discover tests/` couvrant au minimum `replay_buffer`, `termination`, `seeding`, `env_factory` (cf. §9 de `PROJECT.md`).
10. Documentation complète : `README.md` (utilisateur), `PROJECT.md` (hub, figé sauf évolution de contrat), `PLAN.md` (ce fichier, à jour), `DOCUMENTATION.md` (historique complet).
11. Soutenance préparée : capacité à justifier chaque hyperparamètre, chaque choix architectural, chaque résultat d'ablation (cf. §14 de `PROJECT.md`).

---

## 📐 Décisions architecturales verrouillées

| Sujet                          | Décision                                                                               | Motivation                                                                                                  |
| ------------------------------ | -------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| Langage                        | Python 3.10+                                                                           | Match les requirements d'`argparse`, `dataclasses`, type hints modernes (`int \| np.ndarray`).              |
| Dépendances RL/ML              | `gymnasium` + `torch` uniquement pour la logique d'apprentissage                       | Contrainte stricte du sujet (§2.1 de `PROJECT.md`).                                                         |
| Utilitaires autorisés          | `numpy`, `pyyaml`, `matplotlib`, `imageio`, `imageio-ffmpeg`                           | Plomberie justifiée par les livrables (YAML, plots, vidéos).                                                |
| Organisation                   | Layout `src/` en modules (§4.1 `PROJECT.md`) + entrypoints `train.py` / `eval.py` fins | Séparation des responsabilités ; ajouter un agent/wrapper/buffer = nouveau fichier, pas d'édit du trainer.  |
| Abstraction Agent              | ABC `Agent` avec `act / observe / learn / save / load`                                 | Trainer agnostique → bonus (Double DQN, PER, PPO) sans toucher `trainer.py`.                                |
| Registre                       | `AGENTS = {"name": Class, ...}` dans `src/agents/registry.py`                          | Ajouter un agent = **une ligne** dans le registre.                                                          |
| Configs                        | YAML → `dataclasses` typés à la frontière `utils/config.py`                            | Fin du magic-number, validation explicite, persistance fidèle dans `runs/<run>/config.yaml`.                |
| Env factory                    | `build_env(cfg, seed, render_mode)` unique ; wrappers composés depuis config           | Changer l'env = une clé YAML ; aucun code ne `gym.make()` ailleurs.                                         |
| Replay buffer                  | Uniform `ReplayBuffer` (numpy-based, ring buffer) dès la Phase 0                       | Bonus PER = nouveau fichier `prioritized_buffer.py` sans refactor.                                          |
| Réseau                         | Factory `build_mlp(cfg)` paramétrée par `hidden_sizes`, `activation`                   | Bonus Dueling / Noisy Nets = `dueling_mlp.py` / `noisy_layers.py` additifs.                                 |
| Seeds                          | `utils/seeding.py` seed : Python / NumPy / Torch (CPU+CUDA) / `env.reset(seed=...)`    | `torch.use_deterministic_algorithms(True)` quand faisable. Seeds `0..4` pour chaque run reporté.            |
| Logging                        | `utils/logger.py` → CSV (`metrics.csv`) + mirror stdout (`events.log`)                 | Colonnes stables entre runs (cf. §8 `PROJECT.md`) → plots diffables.                                        |
| Classification de terminaison  | `utils/termination.py::classify(info, reward, state) → TerminationReason`              | Exigence §3.5 et §7.2 `PROJECT.md` ; testée dans `tests/test_termination.py`.                               |
| Video recording                | `utils/video.py` via `imageio`, appelé par `evaluator.py` selon `eval.record_video`    | Livrable vidéos (§5 `PROJECT.md`) sans tirer `moviepy`.                                                     |
| Repro                          | `reproduce.sh` boucle seeds `0..4` sur chaque config baseline + ablation               | Acceptance criteria §7.3 `PROJECT.md`.                                                                      |
| Tests                          | stdlib `unittest` uniquement                                                           | Pas de nouvelle dépendance (`pytest` n'est pas justifié pour 4 tests).                                      |
| Versioning                     | Schéma `MAJOR.MINOR.PATCH` interne (voir `DOCUMENTATION.md`)                           | MAJOR = passage de phase ; MINOR = feature ; PATCH = fix/refactor/doc.                                      |

---

## 🧱 Signatures d'interfaces prévues (contrat public)

Ces signatures sont la **frontière stable** entre modules. Elles doivent être respectées lors de l'implémentation ; toute évolution de signature est un breaking change documenté dans `DOCUMENTATION.md`.

```python
# src/agents/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass(frozen=True)
class Transition:
    obs: np.ndarray            # shape (8,) float32
    action: int                # discrete action id
    reward: float
    next_obs: np.ndarray       # shape (8,) float32
    terminated: bool
    truncated: bool

class Agent(ABC):
    @abstractmethod
    def act(self, obs: np.ndarray, *, greedy: bool = False) -> int | np.ndarray: ...
    @abstractmethod
    def observe(self, transition: Transition) -> None: ...       # push to buffer
    @abstractmethod
    def learn(self) -> dict[str, float]: ...                     # one grad step; returns metrics
    @abstractmethod
    def save(self, path: Path) -> None: ...
    @abstractmethod
    def load(self, path: Path) -> None: ...
```

```python
# src/env/factory.py
import gymnasium as gym
from src.utils.config import EnvConfig

def build_env(cfg: EnvConfig, *, seed: int, render_mode: str | None = None) -> gym.Env: ...
```

```python
# src/memory/replay_buffer.py
class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape: tuple[int, ...], action_dtype=np.int64): ...
    def push(self, transition: Transition) -> None: ...
    def sample(self, batch_size: int) -> dict[str, torch.Tensor]: ...   # keys: obs, action, reward, next_obs, done
    def __len__(self) -> int: ...
```

```python
# src/utils/termination.py
from enum import Enum
class TerminationReason(str, Enum):
    LAND = "land"
    CRASH = "crash"
    OUT_OF_VIEW = "out_of_view"
    SLEEP = "sleep"
    TRUNCATED = "truncated"
    UNKNOWN = "unknown"

def classify(info: dict, reward: float, state: np.ndarray, terminated: bool, truncated: bool) -> TerminationReason: ...
```

```python
# src/training/trainer.py (squelette — agent-agnostic, env-agnostic)
def train(cfg: RunConfig, *, run_dir: Path) -> dict: ...     # returns summary (mean_return, solved, ...)
```

```python
# src/training/evaluator.py
def evaluate(agent: Agent, env: gym.Env, *, episodes: int, greedy: bool, record_video: bool = False,
             video_dir: Path | None = None) -> dict: ...
```

---

## 🗓️ Phases & tickets

Chaque ticket a : **objectif** · **livrable** · **critère d'acceptation** (vérifiable). Le ticket n'est coché que quand son critère passe ET que `DOCUMENTATION.md` contient son entrée.

---

### Phase 0 — Fondations (avant tout code d'agent)

> But : pouvoir lancer `python train.py --config configs/smoke.yaml --seed 0` et voir le squelette tourner (env instancié, logger créé, dossier `runs/<run>/<seed>/` rempli, zéro agent encore implémenté). Toute l'architecture est posée pour que les phases suivantes n'éditent que leurs propres fichiers.

- [x] **T0.1 — Initialisation du dépôt**
  - **Objectif** : arborescence vide cohérente avec §4.1 de `PROJECT.md`, `.gitignore` strict (`runs/`, `__pycache__/`, `*.pyc`, `.venv/`, `build/`, `*.egg-info`, `assets/` optionnel selon politique commit), `README.md` squelette, `requirements.txt` minimal.
  - **Livrable** : structure complète + fichiers de docs (`PLAN.md`, `DOCUMENTATION.md` déjà créés) + `.gitignore` + `README.md` + `requirements.txt`.
  - **Critère d'acceptation** : `git status` propre après premier commit ; `tree -L 2 -I '__pycache__'` affiche l'arborescence §4.1 ; aucun fichier généré tracké ; `pip install -r requirements.txt` passe sur une `.venv` fraîche.

- [ ] **T0.2 — Setup Python (venv, requirements, Makefile)**
  - **Objectif** : commandes standards `make install`, `make train CONFIG=...`, `make eval RUN=...`, `make test`, `make reproduce`, `make clean` qui routent vers les bons scripts.
  - **Livrable** : `Makefile` racine, `requirements.txt` pinné (versions exactes), éventuel `requirements-dev.txt` (si besoin `black`/`ruff` plus tard — **non obligatoire**).
  - **Critère d'acceptation** : `make install` crée la `.venv` et installe toutes les deps sans erreur ; `python -c "import gymnasium, torch, yaml, matplotlib, imageio"` passe ; `python -c "import gymnasium as gym; gym.make('LunarLander-v3')"` passe.

- [ ] **T0.3 — `utils/seeding.py` (reproductibilité de base)**
  - **Objectif** : fonction `seed_all(seed: int)` qui seed Python `random`, NumPy, Torch (CPU + CUDA), active `torch.use_deterministic_algorithms(True)` quand faisable, documente les sources de non-déterminisme résiduelles.
  - **Livrable** : `src/utils/seeding.py` + `tests/test_seeding.py`.
  - **Critère d'acceptation** : deux appels à `seed_all(42)` suivis d'une boucle de 100 actions uniformes dans un env scellé produisent exactement la même séquence d'obs, rewards, terminations ; le test `tests/test_seeding.py` vérifie cette invariance.

- [ ] **T0.4 — `utils/config.py` (dataclasses typées + YAML loader)**
  - **Objectif** : `@dataclass` hiérarchiques (`EnvConfig`, `WrapperConfig`, `NetworkConfig`, `EpsilonConfig`, `TargetUpdateConfig`, `AgentConfig`, `TrainingConfig`, `EvalConfig`, `RunConfig`) + `load_config(path) → RunConfig` + `save_config(cfg, path)`.
  - **Livrable** : `src/utils/config.py`.
  - **Critère d'acceptation** : `load_config("configs/baseline/dqn.yaml")` retourne un `RunConfig` typé sans erreur ; champ absent → `ValueError` explicite (chemin YAML + clé manquante) ; `save_config(cfg, path)` puis `load_config(path)` → égalité bit-à-bit (roundtrip).

- [ ] **T0.5 — `utils/logger.py` (CSV + stdout mirror)**
  - **Objectif** : `Logger(run_dir: Path, metric_columns: list[str])` qui écrit `metrics.csv` append-only + mirror `events.log` + flush forcé toutes les N lignes. Colonnes **stables** d'après §8 de `PROJECT.md` : `episode, global_step, return, length, epsilon, loss_mean, loss_std, termination_reason, terminated, truncated, wall_time_s`.
  - **Livrable** : `src/utils/logger.py`.
  - **Critère d'acceptation** : 100 appels à `logger.log(episode=…, return=…, …)` produisent un CSV valide ouvrable par pandas (test manuel) ; les colonnes sont dans l'ordre attendu ; `events.log` contient le même contenu sous forme lisible.

- [ ] **T0.6 — `utils/termination.py` + tests synthétiques**
  - **Objectif** : énumération `TerminationReason` + fonction `classify(info, reward, state, terminated, truncated) → TerminationReason` qui distingue `LAND` / `CRASH` / `OUT_OF_VIEW` / `SLEEP` / `TRUNCATED` selon les heuristiques Box2D (legs contact, position hors frame, reward ≈ ±100, flag `truncated`).
  - **Livrable** : `src/utils/termination.py` + `tests/test_termination.py`.
  - **Critère d'acceptation** : le test synthétique valide chaque cas (ex : `(both_legs_contact=1, reward≈+100, terminated=True)` → `LAND` ; `reward≈-100, terminated=True, legs=0` → `CRASH` ; etc.). 0 `UNKNOWN` sur un set de 1000 épisodes aléatoires réels — à valider en T1.2.

- [ ] **T0.7 — `env/factory.py` + `env/wrappers.py`**
  - **Objectif** : `build_env(cfg, seed, render_mode)` unique, qui instancie `gym.make("LunarLander-v3", **cfg.kwargs)` puis compose les wrappers listés dans `cfg.wrappers` (registre `WRAPPERS = {"RecordEpisodeStats": ..., "ObsNormalize": ..., "RewardClip": ...}`). Seed via `env.reset(seed=seed)`.
  - **Livrable** : `src/env/factory.py`, `src/env/wrappers.py`, `tests/test_env_factory.py`.
  - **Critère d'acceptation** : `build_env(cfg, seed=0)` renvoie un `gym.Env` avec `observation_space.shape == (8,)` et `action_space.n == 4` ; le test `test_env_factory.py` vérifie que les wrappers de la config sont bien appliqués dans l'ordre.

- [ ] **T0.8 — `agents/base.py` + `agents/registry.py`**
  - **Objectif** : ABC `Agent` (signatures gelées en §Signatures ci-dessus) + `Transition` dataclass + registre `AGENTS: dict[str, type[Agent]]` initial vide ; décorateur `@register_agent("name")` pour enregistrer proprement.
  - **Livrable** : `src/agents/base.py`, `src/agents/registry.py`.
  - **Critère d'acceptation** : `from src.agents.registry import AGENTS; AGENTS["random"]` lèvera `KeyError` tant que T1.1 n'est pas fait — attendu. `Agent` ne peut pas être instanciée directement (`TypeError: Can't instantiate abstract class`).

- [ ] **T0.9 — `memory/replay_buffer.py` + tests**
  - **Objectif** : `ReplayBuffer(capacity, obs_shape, action_dtype)` implémenté en ring-buffer NumPy ; `push(transition)`, `sample(batch_size) → dict[str, torch.Tensor]` avec clés `obs, action, reward, next_obs, done` (float32/int64 cohérents).
  - **Livrable** : `src/memory/replay_buffer.py`, `tests/test_replay_buffer.py`.
  - **Critère d'acceptation** : test couvre (a) shape des tenseurs échantillonnés, (b) absence de NaN, (c) FIFO eviction après saturation, (d) `len(buffer) == min(capacity, pushed)`.

- [ ] **T0.10 — `networks/mlp.py`**
  - **Objectif** : factory `build_mlp(input_dim, output_dim, hidden_sizes, activation) → torch.nn.Sequential`. Initialisation par défaut : He (`kaiming_uniform_`) pour ReLU.
  - **Livrable** : `src/networks/mlp.py`.
  - **Critère d'acceptation** : `build_mlp(8, 4, [128, 128], "relu")` renvoie un module avec `param count ≈ 8*128 + 128 + 128*128 + 128 + 128*4 + 4` ; `model(torch.zeros(1, 8)).shape == (1, 4)`.

- [ ] **T0.11 — `training/trainer.py` squelette (agent-agnostic)**
  - **Objectif** : boucle de training générique qui ne connaît QUE `Agent` + `gym.Env` + `Logger`. Pas encore d'algo spécifique ; utilisable avec `RandomAgent` dès T1.1 pour tester la chaîne. Gère : reset, step, `agent.observe`, conditionnel `agent.learn()` si buffer suffisant, logging périodique, évaluation périodique, checkpointing.
  - **Livrable** : `src/training/trainer.py`, `train.py` (entrypoint CLI fin : `--config`, `--seed`, `--out-dir`).
  - **Critère d'acceptation** : `python train.py --config configs/smoke.yaml --seed 0` (où `smoke.yaml` utilisera `RandomAgent` en T1.1) crée `runs/smoke/0/{config.yaml, metrics.csv, events.log}` ; zéro crash sur 1000 steps.

- [ ] **T0.12 — `training/evaluator.py` + `eval.py`**
  - **Objectif** : fonction `evaluate(agent, env, episodes, greedy, record_video) → dict` qui renvoie `mean_return`, `std_return`, `ci95`, `termination_breakdown`, `solved (bool)` + optionnellement enregistre une vidéo par `eval.video_every` épisodes.
  - **Livrable** : `src/training/evaluator.py`, `eval.py` (entrypoint CLI : `--run`, `--episodes`, `--greedy`, `--record-video`).
  - **Critère d'acceptation** : `python eval.py --run runs/smoke/0 --episodes 10 --greedy` produit `runs/smoke/0/eval_<timestamp>.json` avec les champs attendus ; zéro crash sur agent `RandomAgent`.

- [ ] **T0.13 — `utils/video.py` + `utils/plotting.py` (stubs)**
  - **Objectif** : `save_video(frames, path)` via `imageio` ; `plot_returns(csv_path, out_path)`, `plot_loss(...)`, `plot_epsilon(...)`, `plot_termination_breakdown(...)`. Implémentation minimale — les plots détaillés viendront avec les runs réels.
  - **Livrable** : `src/utils/video.py`, `src/utils/plotting.py`.
  - **Critère d'acceptation** : `save_video([np.zeros((400, 600, 3), dtype=np.uint8)] * 30, "test.mp4")` produit un MP4 lisible ; `plot_returns(csv, png)` produit un PNG non-vide sur un metrics.csv factice.

- [ ] **T0.14 — Config smoke + run end-to-end**
  - **Objectif** : `configs/smoke.yaml` minimal qui fait tourner le trainer avec un agent no-op (en attendant T1.1) ; valide toute la chaîne Phase 0.
  - **Livrable** : `configs/smoke.yaml`.
  - **Critère d'acceptation** : `python train.py --config configs/smoke.yaml --seed 0` s'exécute 5000 steps sans crash, produit le dossier `runs/smoke/0/` complet, le test `python -m unittest discover tests/` passe en vert.

**🏁 Fin de Phase 0** : toute l'architecture est en place, la chaîne CLI end-to-end tourne avec un agent factice, les 4 tests unitaires obligatoires passent.

---

### Phase 1 — Baselines & Instrumentation (§6.1 de `PROJECT.md`)

> But : produire deux agents de référence (random + heuristique), valider que la classification de terminaison et l'enregistrement vidéo fonctionnent sur des trajectoires réelles, obtenir les assets "random" et "heuristic" pour le rapport.

- [ ] **T1.1 — `agents/random_agent.py`**
  - **Objectif** : `RandomAgent` qui tire uniformément dans `action_space`. Implémente toutes les méthodes ABC (learn = no-op, save/load = dump/load d'un seed).
  - **Livrable** : `src/agents/random_agent.py` + entrée dans `AGENTS` + `configs/baseline/random.yaml`.
  - **Critère d'acceptation** : `python train.py --config configs/baseline/random.yaml --seed 0` produit un `runs/random/0/` avec `mean_return ≈ -150` (ordre de grandeur attendu d'une politique aléatoire sur LunarLander) ; vidéo sauvegardée.

- [ ] **T1.2 — `agents/heuristic_agent.py`**
  - **Objectif** : politique PD-style sur (`theta`, `theta_dot`, `v_y`, `x`) — typiquement : si penchant à gauche/droite → fire side thruster opposé ; si `v_y` trop négatif → fire main ; sinon do nothing. Formule et coefficients documentés dans le docstring et dans `DOCUMENTATION.md`.
  - **Livrable** : `src/agents/heuristic_agent.py` + entrée dans `AGENTS` + `configs/baseline/heuristic.yaml`.
  - **Critère d'acceptation** : `mean_return ≥ 0` sur 100 épisodes greedy (une heuristique raisonnable bat nettement random sans atteindre forcément 200) ; termination_breakdown montre une proportion `LAND` visible.

- [ ] **T1.3 — Validation de `termination.classify` sur trajectoires réelles**
  - **Objectif** : confirmer que sur un run de `RandomAgent` et `HeuristicAgent`, la proportion de `UNKNOWN` est **0 %** et que la breakdown est plausible (random → majorité `CRASH`/`OUT_OF_VIEW`, heuristic → fraction non nulle de `LAND`).
  - **Livrable** : assert runtime dans l'evaluator ou script de validation dédié (`scripts/validate_termination.py`).
  - **Critère d'acceptation** : `UNKNOWN == 0` sur 100 épisodes random + 100 épisodes heuristic ; log écrit dans `events.log`.

- [ ] **T1.4 — Assets vidéos des baselines**
  - **Objectif** : extraire une vidéo MP4 propre (≤ 30 s) de chaque baseline pour le rapport.
  - **Livrable** : `assets/videos/random.mp4`, `assets/videos/heuristic.mp4`.
  - **Critère d'acceptation** : les deux vidéos sont visionnables, ≤ 5 MB chacune, et leur génération est reproductible via `python eval.py --run runs/<baseline>/<seed> --record-video`.

**🏁 Fin de Phase 1** : baselines fonctionnelles, termination classifier validé sur trajectoires réelles, 2 vidéos commitées dans `assets/`.

---

### Phase 2 — DQN MVP (§6.2 de `PROJECT.md`)

> But : implémenter le DQN de référence, entraîner un agent qui atteint l'objectif ≥ 200 sur 100 épisodes greedy, obtenir une première batch de plots de training.

- [ ] **T2.1 — `agents/dqn.py` : squelette (forward Q + action selection)**
  - **Objectif** : classe `DQNAgent(AgentConfig)` qui construit `q_net` + `target_net` via `build_mlp`, implémente `act(obs, greedy)` avec ε-greedy (ε calculé par `training/schedules.py`), `observe()` pousse dans le buffer.
  - **Livrable** : `src/agents/dqn.py`, `src/training/schedules.py` (linear epsilon decay).
  - **Critère d'acceptation** : `python -c "from src.agents.registry import AGENTS; AGENTS['dqn'](...)"` instancie l'agent sans erreur ; `act(obs, greedy=True)` renvoie `argmax(Q(obs))` ; `act(obs, greedy=False)` mélange exploration et exploitation selon ε courant.

- [ ] **T2.2 — `agents/dqn.py` : `learn()` (bootstrap + target update)**
  - **Objectif** : sample du buffer, calcul de `y = r + γ * max_a' Q_target(s', a') * (1 - done)`, loss MSE ou Huber, backward + Adam step, gradient clipping, target update (hard copy tous les N steps).
  - **Livrable** : `agents/dqn.py` mis à jour.
  - **Critère d'acceptation** : `learn()` renvoie `{"loss": float, "q_mean": float, "td_error_mean": float}` ; 1000 steps d'apprentissage sur le buffer pré-rempli donnent une loss qui décroît mesurablement.

- [ ] **T2.3 — `configs/baseline/dqn.yaml` (référence)**
  - **Objectif** : config complète (hidden_sizes [128,128], γ=0.99, lr=5e-4, batch=64, buffer=100000, min_buffer=1000, target hard every 1000 steps, ε 1.0 → 0.05 sur 100k steps, grad_clip=10.0, 500k total steps, eval every 10k).
  - **Livrable** : `configs/baseline/dqn.yaml`.
  - **Critère d'acceptation** : `load_config("configs/baseline/dqn.yaml")` passe ; aucun warning sur champs inconnus ou manquants.

- [ ] **T2.4 — Training run single-seed + premier jalon ≥ 200**
  - **Objectif** : `python train.py --config configs/baseline/dqn.yaml --seed 0` tourne jusqu'à la fin, l'évaluation finale 100 épisodes greedy donne `mean_return ≥ 200`. Durée target : **≤ 2 h sur CPU** (laptop moderne, si > 2 h prévoir GPU / réduction du total_steps).
  - **Livrable** : `runs/dqn_baseline/0/` complet (config + metrics + checkpoints + events + summary.json).
  - **Critère d'acceptation** : `runs/dqn_baseline/0/summary.json` contient `"solved": true` et `"mean_return_last_100" ≥ 200`. Ceci est le jalon MVP.

- [ ] **T2.5 — Plots de training (returns, loss, epsilon)**
  - **Objectif** : utiliser `utils/plotting.py` pour générer `assets/plots/returns_per_episode.png`, `loss.png`, `epsilon.png`, `episode_length.png`, `termination_breakdown.png` depuis `runs/dqn_baseline/0/metrics.csv`.
  - **Livrable** : 5 PNG dans `assets/plots/`.
  - **Critère d'acceptation** : les 5 plots existent, sont lisibles, avec titres + axes labellés + grille.

- [ ] **T2.6 — Vidéo du DQN entraîné**
  - **Objectif** : `python eval.py --run runs/dqn_baseline/0 --episodes 5 --greedy --record-video`.
  - **Livrable** : `assets/videos/dqn_trained.mp4`.
  - **Critère d'acceptation** : vidéo visionnable, ≤ 5 MB, montre au moins un atterrissage propre (legs contact + lander arrêté sur plateforme).

**🏁 Fin de Phase 2** : **MVP validé au seuil ≥ 200**. Plots de training + vidéo DQN entraîné commitées.

---

### Phase 3 — Ablations & Reproductibilité (§6.3 + §6.4 de `PROJECT.md`)

> But : reproduire la baseline DQN sur 5 seeds, mener au moins 2 ablations, produire les paired-plots et intervalles de confiance à 95 %, mettre en place `reproduce.sh` complet.

- [ ] **T3.1 — Runs multi-seeds de la baseline DQN (seeds 0..4)**
  - **Objectif** : lancer 5 trainings avec seeds 0, 1, 2, 3, 4. Agrégation : mean ± 95 % CI sur `mean_return_last_100` et `first_solve_step`.
  - **Livrable** : `runs/dqn_baseline/{0,1,2,3,4}/` + `runs/dqn_baseline/summary_aggregated.json`.
  - **Critère d'acceptation** : au moins 4 seeds sur 5 ont `solved=true` ; CI95 du mean_return calculé et commité.

- [ ] **T3.2 — `utils/metrics.py` (aggregation, CI95)**
  - **Objectif** : helpers `mean_ci95(values) → (mean, lo, hi)`, `aggregate_runs(pattern) → dict` utilisés par T3.1 et par la génération de plots paired.
  - **Livrable** : `src/utils/metrics.py`.
  - **Critère d'acceptation** : test manuel `mean_ci95([1,2,3,4,5])` donne un CI cohérent ; intégré dans la génération de `summary_aggregated.json`.

- [ ] **T3.3 — Ablation A (choix à arrêter dans `DOCUMENTATION.md`)**
  - **Objectif** : un seul paramètre changé par rapport à baseline. Recommandé : **buffer size** (1e4 vs 1e5 vs 1e6) OU **target update hard vs soft (Polyak τ)**.
  - **Livrable** : `configs/ablations/dqn_<ablation_A>.yaml` + runs seeds 0..4 + paired plot vs baseline.
  - **Critère d'acceptation** : paired plot commité + paragraphe d'analyse rédigé dans le brouillon de rapport.

- [ ] **T3.4 — Ablation B (différente de A)**
  - **Objectif** : second paramètre. Recommandé : **obs normalization** ON/OFF OU **reward clipping** ON/OFF OU **depth réseau** (2 vs 3 hidden layers).
  - **Livrable** : `configs/ablations/dqn_<ablation_B>.yaml` + runs seeds 0..4 + paired plot.
  - **Critère d'acceptation** : idem T3.3.

- [ ] **T3.5 — `reproduce.sh` end-to-end**
  - **Objectif** : script bash qui lance tous les trainings commitée (baseline + 2 ablations × 5 seeds = 15 runs), régénère tous les plots et tous les résumés.
  - **Livrable** : `reproduce.sh` à la racine.
  - **Critère d'acceptation** : `bash reproduce.sh` sur une machine propre régénère l'intégralité du dossier `assets/plots/` et de `runs/` avec des valeurs à **±5 %** des figures commitées ; documenté dans le README.

- [ ] **T3.6 — Plots agrégés (returns avec CI95 shading)**
  - **Objectif** : `plot_returns_multi_seed(run_dirs, out_path)` qui trace mean + shaded CI95 ; réutilisé par baseline et par les paired-plots d'ablation.
  - **Livrable** : `src/utils/plotting.py` enrichi.
  - **Critère d'acceptation** : 3 plots agrégés commitées (baseline, ablation A, ablation B) + 2 paired-plots (A vs baseline, B vs baseline).

**🏁 Fin de Phase 3** : reproductibilité §7.3 validée ; 2 ablations chiffrées + plots agrégés ; 15 runs totaux commitées (ou leurs summaries + metrics CSV — pas les checkpoints .pt volumineux).

---

### Phase 4 — Livrables finaux (§5 + §6.5 de `PROJECT.md`)

> But : assembler toutes les pièces demandées par le sujet : README utilisateur complet, rapport PDF, vidéos showcase, préparation de la soutenance.

- [ ] **T4.1 — `README.md` utilisateur**
  - **Objectif** : sections « Qu'est-ce que c'est », « Installation », « Commandes principales », « Structure du dépôt », « Reproduire les résultats », « Résultats principaux » (avec 2-3 chiffres clés), « Tests », « Crédits ».
  - **Livrable** : `README.md` à la racine.
  - **Critère d'acceptation** : quelqu'un qui clone à froid peut installer et lancer un training en suivant uniquement le README. Relecture croisée effectuée.

- [ ] **T4.2 — Rapport PDF (4-6 pages)**
  - **Objectif** : rédaction du rapport couvrant (a) setup, (b) algorithme, (c) résultats (plot final + table CI95), (d) au moins une ablation analysée, (e) termination breakdown, (f) limites, (g) next steps. Format PDF.
  - **Livrable** : `report/report.pdf`.
  - **Critère d'acceptation** : 4 à 6 pages, toutes les figures réutilisent les PNG de `assets/plots/`, aucune valeur chiffrée non traçable à un run committé.

- [ ] **T4.3 — Vidéo showcase (trained, slow-motion optionnel)**
  - **Objectif** : vidéo MP4 propre de 20-30 secondes montrant un atterrissage réussi du DQN entraîné (avec potentiellement incrustation du return).
  - **Livrable** : `assets/videos/showcase.mp4`.
  - **Critère d'acceptation** : vidéo visionnable, ≤ 10 MB.

- [ ] **T4.4 — Préparation soutenance**
  - **Objectif** : capacité à répondre aux 7 questions §14 de `PROJECT.md`. Prise de notes dans `report/defense_notes.md` (privé, non livré à l'école mais utile).
  - **Livrable** : slides ou équivalent libre (PDF, HTML, Markdown projeté).
  - **Critère d'acceptation** : auto-vérification des 7 questions. Un collaborateur externe peut rejouer une des démos en < 2 minutes.

**🏁 Fin de Phase 4** : MVP complet, livrable, soutenable.

---

### Phase 5 — Bonus (§12 de `PROJECT.md`)

> But : empiler des bonus algorithmiques et/ou environnementaux. Règle d'or §12.4 : aucune édition de `trainer.py`, `evaluator.py`, `agents/base.py` sauf si un bonus expose une abstraction réellement manquante — auquel cas `PROJECT.md` est mis à jour dans le même patch.

- [ ] **T5.1 — Double DQN** → `src/agents/double_dqn.py` + `configs/bonus/double_dqn.yaml`.
- [ ] **T5.2 — Dueling DQN** → `src/networks/dueling_mlp.py` + clé config dans AgentConfig ; compatible avec DQN et Double DQN existants.
- [ ] **T5.3 — Prioritized Experience Replay (PER)** → `src/memory/prioritized_buffer.py` ; la factory de buffer lit le type depuis la config.
- [ ] **T5.4 — N-step returns** → `src/memory/n_step_wrapper.py` qui wrap un buffer existant.
- [ ] **T5.5 — Noisy Nets** → `src/networks/noisy_layers.py` ; remplace ε-greedy quand activé.
- [ ] **T5.6 — PPO (on-policy, policy gradient)** → `src/agents/ppo.py` + `src/memory/rollout_buffer.py` + `configs/bonus/ppo.yaml`.
- [ ] **T5.7 — SAC (continuous)** → `src/agents/sac.py` + `configs/bonus/sac_continuous.yaml` ; nécessite `env.kwargs.continuous = True`.
- [ ] **T5.8 — Wind** → `configs/bonus/dqn_wind.yaml` avec `enable_wind=True, wind_power=15.0, turbulence_power=1.5` ; run seeds 0..4 + plot de comparaison avec baseline sans vent.
- [ ] **T5.9 — Multi-seed parallel driver** → `scripts/parallel_train.py` utilisant `multiprocessing`, pas de nouvelle dépendance.
- [ ] **T5.10 — Comparative rollout video** → script qui rend côte à côte random / heuristic / trained en une seule vidéo.
- [ ] **T5.11 — CI GitHub Actions** → `.github/workflows/tests.yml` qui lance `python -m unittest discover tests/` sur Ubuntu.

Chaque bonus commité déclenche une sous-section « Bonus X » dans le rapport + une entrée dans `DOCUMENTATION.md`.

---

### Phase 6 — Polish & soutenance finale

- [ ] **T6.1 — Revue finale du code**
  - **Objectif** : zéro code mort, zéro `print` non dans les entrypoints, toutes les fonctions publiques ont un docstring, imports triés, lignes ≤ 100 col.
  - **Livrable** : PR/commit de cleanup.
  - **Critère d'acceptation** : `grep -rn "TODO\|FIXME\|XXX"` dans `src/` donne un résultat vide ou justifié ; revue manuelle par l'auteur.

- [ ] **T6.2 — README final enrichi (résultats + screenshots)**
  - **Objectif** : README avec captures de plots et lien vers la showcase.
  - **Livrable** : `README.md` finalisé avec tous les assets référencés.
  - **Critère d'acceptation** : les liens relatifs vers `assets/plots/*.png` et `assets/videos/*.mp4` fonctionnent sur GitHub.

- [ ] **T6.3 — Rapport PDF final (re-relu, formaté, numéroté)**
  - **Objectif** : passe finale de relecture ; vérifier chiffres = runs committés ; figures haute-résolution.
  - **Livrable** : `report/report.pdf` final.
  - **Critère d'acceptation** : lu par une personne externe au projet sans objection majeure.

- [ ] **T6.4 — Dry-run de la soutenance**
  - **Objectif** : répétition timée (≈ 10 min), scripts prêts à démarrer en local.
  - **Livrable** : notes de défense privées, scénario de démo testé.
  - **Critère d'acceptation** : la démo tient dans le temps imparti ; fallback vidéo prêt au cas où une démo live échouerait.

**🏁 Fin de Phase 6** : projet prêt à être rendu et soutenu.

---

## 📊 Checklist de conformité permanente (à vérifier à chaque commit)

- [ ] Langage Python uniquement, aucune lib RL/ML hors `gymnasium` + `torch` dans le code métier.
- [ ] Aucune lib de tuning (`optuna`, `wandb sweeps`, etc.).
- [ ] `requirements.txt` à jour et pinné.
- [ ] `train.py` et `eval.py` restent des thin CLIs (pas de logique qui dépend du config ailleurs).
- [ ] Colonnes `metrics.csv` stables (§8 `PROJECT.md`).
- [ ] Seeds reproductibles (§10 `PROJECT.md`).
- [ ] `.gitignore` strict : `runs/`, `__pycache__/`, `.venv/`, `*.pyc`, fichiers lourds.
- [ ] `PLAN.md` à jour (cases cochées pour chaque ticket terminé).
- [ ] `DOCUMENTATION.md` à jour (entrée `vX.Y.Z` par ticket).
- [ ] `python -m unittest discover tests/` vert à chaque commit.

---

## 🔁 Processus à chaque ticket

1. Vérifier le ticket dans `PLAN.md` (phase courante, critère d'acceptation lu).
2. Implémenter strictement ce que demande le ticket, sans anticiper un ticket ultérieur.
3. Valider le critère d'acceptation (commande copiée du ticket → résultat attendu).
4. Lancer `python -m unittest discover tests/` → vert.
5. Cocher la case du ticket dans `PLAN.md`.
6. Ajouter une entrée `vX.Y.Z` dans `DOCUMENTATION.md` suivant le format documenté (Résumé, Contexte, Détails techniques, Alternatives rejetées, Impact, Non couvert, Validation).
7. Commit avec message conventionnel : `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`, `perf:`.

---

*Aligné sur `PROJECT.md v1.0.0` (sujet Epitech AIA-401 v0.0.1).*
