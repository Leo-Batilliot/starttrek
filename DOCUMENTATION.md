# DOCUMENTATION.md — Historique du projet Start Trek (Lunar Lander RL)

> Ce document est un **journal technique vivant**. À chaque feature implémentée, à chaque choix notable, à chaque patch, on ajoute une entrée.
>
> - `PROJECT.md` = hub descriptif (sujet, contraintes dures, MVP, livrables, bonus).
> - `PLAN.md` = feuille de route (tickets, phases, critères d'acceptation).
> - `DOCUMENTATION.md` (ce fichier) = **ce qui a été fait, quand, pourquoi, comment**.
>
> **Règle** : on ne coche pas un ticket dans `PLAN.md` sans avoir ajouté son entrée ici.

---

## 📑 Sommaire

- [Format d'une entrée](#format-dune-entrée)
- [Versioning](#versioning)
- [Changelog](#changelog)
- [Architecture & choix techniques](#architecture--choix-techniques)
- [Features documentées](#features-documentées)
- [Patches / bug fixes](#patches--bug-fixes)
- [Known issues](#known-issues)

---

## Format d'une entrée

Chaque entrée suit ce canevas :

```markdown
### vX.Y.Z — YYYY-MM-DD — <Titre court>

**Type** : feature | refactor | fix | docs | infra | perf | test | chore
**Ticket** : TX.Y (référence vers PLAN.md)
**Phase** : 0 (Fondations) | 1 (Baselines) | 2 (DQN MVP) | 3 (Ablations & Repro) | 4 (Livrables) | 5 (Bonus) | 6 (Polish)

**Résumé**
Une phrase qui dit ce qui change.

**Contexte / motivation**
Pourquoi on fait ça maintenant, quelle contrainte ou quelle feature ça débloque.

**Détails techniques**
- Fichiers ajoutés / modifiés / supprimés
- Interfaces ou signatures impactées
- Dépendances introduites
- Décisions notables

**Alternatives rejetées**
Ce qu'on aurait pu faire et pourquoi on ne l'a pas fait. Important : les mauvaises idées écartées sont souvent plus instructives que la bonne retenue.

**Impact**
- Ce qui devient possible
- Ce qui n'est PAS couvert (volontairement)
- Risques / dettes techniques introduites

**Non couvert volontairement**
Ce que ce ticket ne fait PAS exprès, avec pointeur vers le ticket qui le fera.

**Validation**
- Commande(s) permettant de vérifier que ça marche
- Valeurs numériques attendues (ex : `mean_return ≥ 200`)
- Run de référence (ex : `runs/dqn_baseline/0/`)
```

---

## Versioning

Schéma **MAJOR.MINOR.PATCH** interne au projet (pas de release publique) :

- **MAJOR** : passage d'une phase à la suivante (0 → 1 → 2 → … → 6).
- **MINOR** : nouvelle feature complète (un agent, un wrapper, un bonus, un livrable majeur).
- **PATCH** : bug fix, refactor, amélioration de doc, ajout de scène/config mineure.

Exemples de jalons attendus :
- `v0.1.0` = Phase 0 démarrée, première brique d'infra livrée.
- `v1.0.0` = Phase 1 validée (baselines fonctionnelles).
- `v2.4.0` = DQN atteint ≥ 200 (T2.4 livré).
- `v3.5.0` = `reproduce.sh` end-to-end + 2 ablations closes.
- `v4.0.0` = README + rapport + showcase commitée ; MVP livrable.
- `v5.X.0` = empilement des bonus (5.1, 5.2, ...).
- `v6.0.0` = polish final, projet soutenable.

---

## Changelog

<!-- Ajouter les futures entrées ci-dessous, en ordre antéchronologique (le plus récent en premier). -->

### v0.1.0 — 2026-04-22 — Initialisation du dépôt (T0.1)

**Type** : infra
**Ticket** : T0.1
**Phase** : 0 (Fondations)

**Résumé**
Mise en place de l'arborescence §4.1 de `PROJECT.md` : dossiers `configs/{baseline,ablations,bonus}`, `src/{agents,networks,memory,env,training,utils}`, `assets/{plots,videos}`, `report/`, `tests/`. Ajout des `__init__.py` pour chaque package Python, des `.gitkeep` pour les dossiers sans fichier committé. Écriture d'un `.gitignore` strict (runs, venv, caches Python, artefacts IDE), d'un `requirements.txt` pinné (gymnasium[box2d], torch, numpy, pyyaml, matplotlib, imageio, imageio-ffmpeg) et d'un `README.md` squelette. Suppression du `main.py` template PyCharm qui n'appartient pas au layout du projet.

**Contexte / motivation**
Premier ticket concret du plan. Le but n'est pas d'écrire du code métier, mais de **poser les rails physiques** du dépôt pour que tous les tickets ultérieurs sachent où déposer leurs fichiers sans réfléchir. `PROJECT.md` §4.1 fixe un layout précis ; le respecter dès maintenant évite :
- les imports cassés dus à des `src/` créés a posteriori sans `__init__.py` ;
- les commits accidentels de binaires, `.venv`, `runs/` volumineux ;
- la dérive "ah tiens où je mets ce fichier".

Le `.gitignore` est particulièrement critique pour un projet RL : un seul run DQN produit facilement 200-500 MB de checkpoints + vidéos — les pousser sur git serait catastrophique. `runs/` est donc **gitignored par défaut** ; les assets dérivés pour le rapport sont sélectivement copiés vers `assets/` par `reproduce.sh` (Phase 3).

**Détails techniques**

- **Arborescence créée** (tous les dossiers §4.1 de `PROJECT.md` sauf ceux qui sont livrables de tickets ultérieurs) :
  ```
  configs/{baseline,ablations,bonus}/.gitkeep
  src/__init__.py
  src/{agents,networks,memory,env,training,utils}/__init__.py
  assets/{plots,videos}/.gitkeep
  report/.gitkeep
  tests/__init__.py
  ```
  `src/` est un package Python (`__init__.py`) pour autoriser `from src.agents.base import Agent` depuis `train.py`. Les sous-modules sont aussi des packages → imports propres, pas de bricolage `sys.path`.
- **Fichiers NON créés (intentionnellement)** :
  - `train.py` → ticket **T0.11** (trainer squelette).
  - `eval.py` → ticket **T0.12** (evaluator + CLI).
  - `reproduce.sh` → ticket **T3.5** (quand les configs à lancer sont connues).
  - `report/report.pdf` → ticket **T4.2** (rédaction).
  - Les agents / réseaux / buffers / wrappers → leurs tickets respectifs (T0.8, T0.9, T0.10, T0.7, Phase 1+).
- **`.gitignore`** (réécrit — la version initiale `*.` + `!.gitignore` de l'utilisateur était trop large et pas spécialisée Python) :
  - **Python** : `__pycache__/`, `*.py[cod]`, `*.egg-info/`, `dist/`, `build/`, `.Python`.
  - **Virtual envs** : `.venv/`, `venv/`, `env/`, `.env`.
  - **Training artifacts** : `runs/` (le gros morceau — gitignored pour ne pas pousser des GB de checkpoints).
  - **IDE** : `.idea/` (ajouté — le dossier PyCharm de l'utilisateur est généré et personnel), `.vscode/`, `*.swp`, `.DS_Store`, `Thumbs.db`.
  - **Coverage / tests** : `.coverage`, `.pytest_cache/`, `htmlcov/`.
  - **Torch artifacts hors runs** : `*.pt`, `*.pth`, `*.ckpt` avec whitelist `!assets/**/*.pt` (au cas où un checkpoint "showcase" serait commité en Phase 4).
  - **Logs bruts** : `*.log`.
- **`requirements.txt`** — versions pinnées exactes :
  ```
  gymnasium[box2d]==1.0.0
  torch==2.5.1
  numpy==2.1.3
  pyyaml==6.0.2
  matplotlib==3.9.2
  imageio==2.36.1
  imageio-ffmpeg==0.5.1
  ```
  Choix des versions : dernières stables mi-2025, compatibles entre elles (torch 2.5.x + numpy 2.1.x est un combo validé). `gymnasium[box2d]` tire `box2d-py` qui nécessite `swig` système — documenté dans le `README.md`. Pas de `optuna` / `wandb` / `stable-baselines3` / etc., conformément à §2.1 de `PROJECT.md`.
- **`README.md`** — squelette minimal :
  - Pitch + lien vers les trois docs (`PROJECT.md`, `PLAN.md`, `DOCUMENTATION.md`).
  - Objectif chiffré (mean return ≥ 200 sur 100 épisodes greedy).
  - Installation (venv + pip).
  - Commandes principales (les scripts sont annotés "seront introduits" — honnête sur l'état du projet).
  - Structure du dépôt (reproduction de l'arborescence commentée).
  - Statut (cases cochées). Sera enrichi en T4.1 puis T6.2 avec résultats + captures.
- **Suppression de `main.py`** : c'était le template PyCharm `def print_hi(name): print(f'Hi, {name}')`. Il sera remplacé par `train.py` + `eval.py` aux tickets T0.11 / T0.12 — le garder serait trompeur.

**Alternatives rejetées**

- **Créer `train.py` / `eval.py` / `reproduce.sh` en placeholders vides dès T0.1** : tentant pour que l'arborescence "paraisse complète", mais ça crée des scripts qui ne font rien et qui peuvent induire en erreur (un collaborateur qui les lance obtient un silence). Mieux vaut les absent·e·s jusqu'à leur vrai ticket.
- **Ne pas créer les `__init__.py` (paquets implicites)** : Python 3.3+ supporte les namespace packages sans `__init__.py`, mais les outils de tests (`unittest discover`) et d'import relatif sont plus fiables avec des packages explicites. Le coût est un fichier vide par dossier.
- **Ignorer `.idea/` par commit de suppression (`git rm -r`)** : il n'a jamais été committé (status initial `?? .idea/`). L'ajouter simplement au `.gitignore` suffit.
- **Piner les versions avec `>=` au lieu de `==`** : `PROJECT.md` §10 exige « Pin dependency versions in `requirements.txt` ». Les `>=` rendent la reproductibilité §7.3 impossible à garantir (un `pip install` dans 6 mois tirerait une version différente). Pin strict requis.
- **Utiliser `poetry` ou `pdm` à la place de `requirements.txt` plat** : surdimensionné pour un projet à 6 dépendances ; `PROJECT.md` parle de `requirements.txt` explicitement, on s'y tient.
- **Installer `pytest` au lieu de `unittest`** : `unittest` est dans la stdlib, zéro dépendance supplémentaire. Pour les 4 tests minimaux exigés par §9, c'est largement suffisant et c'est la voie choisie dans `PLAN.md`.
- **Commiter un `configs/baseline/random.yaml` minimal dès T0.1** : tentant pour préparer la route, mais la dataclass `RunConfig` n'existe pas encore (elle vient en T0.4). Un YAML non testable maintenant serait du vaporware.
- **Créer `runs/` avec un `.gitkeep`** : non — le dossier est gitignored, un `.gitkeep` dedans serait paradoxal. Il sera créé automatiquement par `train.py` au premier run (T0.11).

**Impact**
- **T0.2** (Makefile + venv) : peut partir d'un `requirements.txt` valide, cibler des commandes qui n'auront besoin que de `pip install` + les scripts à venir.
- **T0.3+ (toute la Phase 0)** : chaque fichier à écrire a déjà un dossier d'accueil et un package Python parent.
- **T3.5** (`reproduce.sh`) : `runs/` étant gitignored, le script peut reconstruire l'état sans conflits git.
- **Dette technique** : aucune. Les choix sont réversibles (une version peut être bumpée, un dossier peut être renommé avec `git mv` + ajustement des imports).

**Non couvert volontairement**
- **Makefile / cibles `make install|train|eval|test|reproduce|clean`** → T0.2.
- **Validation réelle** `pip install -r requirements.txt` sur `.venv` fraîche → ne peut être faite que sur une machine propre ; le critère est consigné, à valider par l'utilisateur à la prochaine session (ou déjà vérifié si une `.venv` existe).
- **Commitage de `.idea/` ou de son suppression** : l'utilisateur a PyCharm ouvert, `.idea/` est local ; il est ignoré au commit près mais non supprimé du FS. Voir `known issues`.

**Validation**
```bash
# Arborescence conforme à §4.1 de PROJECT.md
find . -not -path './.git*' -not -path './.idea*' -not -path './.claude*' -not -path './other_project*' \
  | grep -E '(configs|src|assets|report|tests)' | sort
# → 23 entrées attendues (7 dossiers src/ + 3 configs/ + 2 assets/ + report + tests + leurs .gitkeep/init)

# Aucun fichier généré tracké
git status --short
# → seuls les fichiers sources (docs + requirements + README + .gitignore + arbo) sont listés

# Packages Python importables dès qu'une venv sera prête
python -c "import src"                                    # doit passer sans erreur
python -c "from src.agents import __init__"              # idem

# (À valider par l'utilisateur sur venv fraîche)
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```
Les 3 premiers checks passent localement. Le 4ᵉ (install) est un critère d'acceptation à valider par l'utilisateur à la prochaine session.

---

<!-- Prochaines entrées à ajouter ci-dessus cette ligne, en ordre antéchronologique. -->

---

## Architecture & choix techniques

Cette section consolide les **décisions architecturales** (WHY plus que WHAT). Chaque décision pointe vers la version où elle a été introduite, pour que l'historique reste traçable.

### Stack et périmètre (v0.0.1)

- **Python 3.10+**. Les features utilisées (`X | Y` pour les types unions, `dataclasses(kw_only=...)`, pattern matching) assument cette version minimum.
- **Dépendances RL/ML** : `gymnasium` + `torch` **exclusivement** pour la logique d'apprentissage (contrainte §2.1 de `PROJECT.md`). Aucune lib comme `stable-baselines3`, `ray[rllib]`, `cleanrl`, `tianshou`, `keras`, `tensorflow`, `jax`.
- **Utilitaires autorisés** : `numpy` (transitif torch/gymnasium), `pyyaml` (YAML configs), `matplotlib` (plots), `imageio` + `imageio-ffmpeg` (vidéos). Chaque ajout est justifié par un livrable explicite.
- **Interdits** : toute lib de tuning automatique (`optuna`, `wandb sweeps`, `hyperopt`) — grid search manuel uniquement.

### Patterns architecturaux (v0.0.1)

- **Interface / Abstract Base Class** — `Agent` (ABC) dans `src/agents/base.py` : contrat unique respecté par Random, Heuristic, DQN, Double DQN, PPO, SAC, etc. Le trainer ne connaît que `Agent` + `gym.Env`.
- **Registry** — `AGENTS: dict[str, type[Agent]]` + décorateur `@register_agent("name")`. Ajouter un agent = un fichier + une ligne.
- **Factory** — `build_env(cfg)`, `build_mlp(cfg)`, futur `build_buffer(cfg)`. Tout ce qui est configurable passe par une factory qui lit la config.
- **Config-as-data** — chaque run est **entièrement** décrit par un fichier YAML. Aucun `if mode == "dqn"` dans `train.py`. Le config dicte.
- **Strategy** (implicite via polymorphisme) — les agents sont des stratégies interchangeables au sein du trainer.
- **Immutabilité post-construction** — `RunConfig` est `frozen=True` ; une fois chargé, il est persisté dans `runs/<run>/config.yaml` et jamais muté.

### Reproductibilité (v0.0.1)

- **Seeds propagés partout** : Python `random`, NumPy, Torch (CPU + CUDA), `env.reset(seed=...)` via `utils/seeding.py::seed_all()`.
- **`torch.use_deterministic_algorithms(True)`** activé quand c'est faisable ; les sources de non-déterminisme résiduelles (CUDA cuDNN benchmarking, ordre des threads Box2D) sont documentées dans les docstrings concernés.
- **Config persistée** : chaque run sérialise sa config exacte dans `runs/<run>/<seed>/config.yaml` — on peut toujours rejouer à partir de cette trace, même si le YAML source a été modifié depuis.
- **Seeds reportés** : 0, 1, 2, 3, 4 pour chaque run présenté dans le rapport (§6.4 de `PROJECT.md`).
- **`reproduce.sh`** à la racine régénère tout depuis les configs + seeds commitées (acceptance criteria §7.3).

### Logging & métriques (v0.0.1)

- **Colonnes CSV stables** pour permettre les plots inter-runs :
  ```
  episode, global_step, return, length, epsilon, loss_mean, loss_std,
  termination_reason, terminated, truncated, wall_time_s
  ```
- **Une ligne par épisode** dans `metrics.csv` ; un `summary.json` par run (mean_return_last_100, ci95, first_solve_step, termination_breakdown).
- **Mirror stdout → `events.log`** pour que ce qui est loggué à l'écran soit aussi conservé sur disque.
- **Pas de TensorBoard** dans le MVP (optionnel en bonus T5.x) pour garder les dépendances minimales.

### Classification des terminaisons (v0.0.1)

- **Exigence sujet** (§3.5 + §7.2 de `PROJECT.md`) : distinguer **crash / land / out_of_view / sleep / truncated**.
- **Implémentation** : `utils/termination.py::classify(info, reward, state, terminated, truncated) → TerminationReason` (enum). Heuristiques :
  - `truncated=True` → `TRUNCATED`.
  - `terminated=True` + `reward ≈ +100` + legs contacts → `LAND`.
  - `terminated=True` + `reward ≈ -100` + corps en contact → `CRASH`.
  - `terminated=True` + `|x| > frame_limit` → `OUT_OF_VIEW`.
  - `terminated=True` + vélocité ~0 + pas de legs contact → `SLEEP` (Box2D asleep).
  - Sinon → `UNKNOWN` (doit être 0 sur trajectoires réelles — assertion au test T1.3).
- **Tests** : `tests/test_termination.py` couvre chaque cas avec des `info`/`reward`/`state` synthétiques.

### Layout des données (v0.0.1)

- **Observations** : `np.float32` shape `(8,)` — cf. §3.2 `PROJECT.md`.
- **Actions** : `int` discret `[0, 1, 2, 3]` (MVP discret) ; `np.ndarray` float en continuous bonus.
- **Transitions dans le buffer** : NumPy ring buffer (pas deque) — accès random uniforme O(1), évite les copies coûteuses sur sample. Conversion vers `torch.Tensor` au moment du `sample(batch_size)` seulement.
- **Colors / couleurs** : non applicable (projet headless, pas de rendu autre que `imageio`).

---

## Features documentées

Chaque feature (agent, wrapper, bonus) aura sa propre sous-section, remplie au fur et à mesure.

### Agents

_Aucun agent implémenté pour l'instant._

### Wrappers d'environnement

_Aucun wrapper implémenté pour l'instant._

### Ablations

_Aucune ablation menée pour l'instant._

### Bonus

_Aucun bonus implémenté pour l'instant._

---

## Patches / bug fixes

_Aucun patch pour l'instant._

---

## Known issues

_Aucun problème connu pour l'instant._
