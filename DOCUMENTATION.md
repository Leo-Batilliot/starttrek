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

### v0.13.0 — 2026-05-11 — Vidéo + plotting stubs (T0.13)

**Type** : feature
**Ticket** : T0.13
**Phase** : 0 (Fondations)

**Résumé**
Ajout de `src/utils/video.py::save_video(frames, path, *, fps=30)` (encodage MP4 via `imageio.v2.mimwrite`) et de `src/utils/plotting.py` avec quatre fonctions stubs (`plot_returns`, `plot_loss`, `plot_epsilon`, `plot_termination_breakdown`) qui lisent `metrics.csv` et produisent des PNG. Couverture par 9 tests vidéo + 7 tests plotting (suite totale 124 / 124 OK).

**Contexte / motivation**
Deux livrables prévus par §5 et §8 de `PROJECT.md` dépendent de ces deux modules : les vidéos (random / heuristic / trained — Phase 1 et Phase 2) et les plots de training (`returns_per_episode.png`, `loss.png`, `epsilon.png`, `termination_breakdown.png` — T2.5). `src/training/evaluator.py` (T0.12) référençait déjà `from src.utils.video import save_video` dans une `try/except ImportError`, donc dès ce ticket la capture vidéo de l'évaluateur devient effective sans avoir à toucher au trainer ou à l'evaluator. Le ticket est explicitement marqué « implémentation minimale » dans `PLAN.md` — les enrichissements (CI95 shaded, paired plots) viendront avec T3.6 quand on aura des runs réels.

**Détails techniques**

- **Fichiers créés** :
  - `src/utils/video.py` — `save_video()`.
  - `src/utils/plotting.py` — 4 fonctions `plot_*`, helpers privés `_read_columns`, `_floats`, `_rolling_mean`, `_save_fig`.
  - `tests/test_video.py` — 9 tests (TestSaveVideo).
  - `tests/test_plotting.py` — 7 tests (TestPlotting).
- **`save_video(frames, path, *, fps=30) → Path`** :
  - Encodage via `imageio.v2.mimwrite(str(path), frames, fps=fps, macro_block_size=1)`. Le kwarg `macro_block_size=1` désactive le padding automatique imposé par ffmpeg sur les dimensions non multiples de 16, ce qui permet d'écrire des frames de taille arbitraire (utile pour les tests synthétiques et les renders Gymnasium qui ne sont pas garantis 16-aligned).
  - Validation à la frontière : `fps > 0`, path avec extension, `frames` non vide, premier frame `(H, W, 3)`, toutes les frames identiques en shape, dtype forcé `uint8` (cast `astype(uint8, copy=False)` quand nécessaire).
  - `path.parent.mkdir(parents=True, exist_ok=True)` — l'evaluator passe `runs/<run>/videos/episode_NNNN.mp4` et ce dossier peut ne pas exister.
  - Retourne le `Path` final pour permettre à l'evaluator de le push dans `result["videos"]`.
- **`plotting.py`** :
  - **Backend `Agg` forcé** (`matplotlib.use("Agg")` avant l'import de `pyplot`) — projet headless, aucun `plt.show()`, important pour CI et pour les WSL sans display.
  - Lecture CSV via stdlib `csv.DictReader` + parsing en `np.float64` via `_floats()` (cellules vides → `NaN`). Aucune dépendance à `pandas`.
  - `_read_columns(csv, cols)` : `FileNotFoundError` si CSV absent, `ValueError` si une colonne demandée manque. Fail loud — plus utile qu'un plot vide silencieux.
  - **`plot_returns(csv, out, *, window=100)`** : trace les returns bruts (alpha 0.4) + rolling mean (window=100) + ligne horizontale `y=200` (seuil "solved" §7.1 PROJECT.md). Convolution NumPy (`np.convolve` mode `valid`) — plus simple que pandas rolling et zéro dépendance.
  - **`plot_loss(csv, out)`** : trace `loss_mean` en filtrant les NaN (les early episodes avant `min_buffer` n'ont pas de loss).
  - **`plot_epsilon(csv, out)`** : trace `epsilon` borné `[0, 1.05]`.
  - **`plot_termination_breakdown(csv, out)`** : compte les `termination_reason`, normalise en fraction, bar chart dans l'ordre canonique (`land`, `crash`, `out_of_view`, `sleep`, `truncated`, `unknown`) avec couleurs distinctes et étiquettes numériques au-dessus de chaque barre.
  - Chaque fonction retourne le `Path` du PNG produit et utilise `_save_fig` qui force `tight_layout()`, `dpi=120`, et `plt.close(fig)` (évite la fuite mémoire de figures matplotlib non fermées sur des séries de runs).
- **Tests vidéo (9)** : MP4 écrit + non vide, MP4 relisible par `imageio.mimread`, `mkdir` parent récursif, `[]` raise, shape `(H, W, 4)` raise, frames hétérogènes raise, `fps=0` raise, path sans extension raise, retour `Path` cohérent. Frames synthétiques 32×48×3 uint8 — taille volontairement modeste pour garder les tests rapides (< 1 s chacun).
- **Tests plotting (7)** : chacun des 4 plots écrit un PNG (magic bytes `\x89PNG` vérifiés, taille > 0) ; `mkdir` parent récursif sur `plot_returns` ; CSV absent → `FileNotFoundError` ; colonne manquante → `ValueError`. Synthèse d'un `metrics.csv` de 200 lignes via `csv.DictWriter` avec les 11 colonnes officielles (cf. §8 PROJECT.md).

**Alternatives rejetées**

- **Utiliser `imageio.v3` (interface stream-based)** : plus moderne mais l'API `iio.imwrite(path, ndarray)` requiert un seul tableau 4D `(T, H, W, 3)` ou un générateur via `imopen`. L'API v2 `mimwrite(path, list_of_frames)` est strictement équivalente côté sortie, plus simple à appeler et déjà la voie suggérée par la doc officielle pour les MP4. Pas de gain à migrer.
- **Tirer `moviepy`** : sur-dimensionné. `imageio + imageio-ffmpeg` est déjà dans `requirements.txt` (T0.1) et couvre 100 % du besoin (encoder une liste de frames RGB en MP4). Ajouter `moviepy` introduirait des deps transitives (`decorator`, `proglog`, `tqdm`) non justifiées.
- **Wrapper `RecordVideo` Gymnasium** : déjà rejeté en T0.7 (non couvert volontairement). L'evaluator (T0.12) consomme `env.render()` puis appelle `save_video(frames, path)` — donne contrôle total sur FPS, dossier, nom de fichier, sans dépendre du naming auto-incrémenté du wrapper.
- **Padding automatique des frames à un multiple de 16** plutôt que `macro_block_size=1` : pénalise la qualité (frames étirées) et masque les vraies dimensions du render. `macro_block_size=1` est explicite et documenté chez imageio.
- **Tolérer un `fps` flottant** : `imageio.mimwrite` l'accepte mais la sémantique d'un MP4 à 29.97 fps n'apporte rien ici — strict `int > 0` est plus simple à valider.
- **Utiliser `pandas` pour parser `metrics.csv`** dans `plotting.py` : ajoute une dépendance lourde (~50 MB) pour ce qu'un `csv.DictReader` + `np.asarray` font en 10 lignes. Refusé : `requirements.txt` est intouchable sauf justification forte (§2.1 PROJECT.md).
- **Plot agrégé multi-seed avec CI95 shading** : ticket T3.6 explicite. Ici on reste sur le single-run.
- **`plot_episode_length` (5ᵉ fonction)** : non listée dans T0.13. T2.5 demandera `episode_length.png` mais l'ajout sera trivial (copie de `plot_returns`). Ne pas surimplémenter (règle "no premature abstraction").
- **Backend `module://matplotlib_inline.backend_inline`** ou autre : `Agg` est le bon choix headless universel.
- **Levée d'erreur sur cellules vides** dans `_floats` : ces vides existent légitimement (loss_mean avant `min_buffer`, epsilon dans certains setups). On les mappe à `NaN` et on filtre via `np.isnan` au tracé.
- **Sauvegarder en `.pdf` ou `.svg`** : §5 PROJECT.md liste explicitement les plots en PNG. Pas de débat.

**Impact**
- **T0.12 (Evaluator)** : le bloc `try: from src.utils.video import save_video except ImportError: return None` qui était inerte devient effectif — les vidéos seront effectivement écrites lors d'un `python eval.py --record-video`.
- **T1.4 (Vidéos baselines)** : Random + Heuristic peuvent désormais sortir leurs MP4 sans changer de code.
- **T2.5 (Plots de training DQN)** : les 4 PNG seront générés en exploitant `runs/dqn_baseline/0/metrics.csv` ; il restera à ajouter `plot_episode_length` (copie de `plot_returns`, ~10 lignes).
- **T2.6 / T4.3 (showcase)** : `save_video` accepte directement la liste de frames retournée par `env.render(render_mode="rgb_array")`.
- **T3.6 (plots multi-seeds + CI95)** : enrichira `plot_returns` ou ajoutera `plot_returns_multi_seed` à côté. La structure modulaire (chaque plot = une fonction publique + helpers privés) rend l'extension non-cassante.
- **Aucune dette technique** : aucune dépendance ajoutée, modules auto-contenus, fail loud sur entrées invalides.

**Non couvert volontairement**
- **`plot_episode_length`** → T2.5.
- **CI95 shaded multi-seed** → T3.6.
- **Paired plots ablation A vs baseline** → T3.3 / T3.4 / T3.6.
- **Encoder choice (h264 / h265 / mp4v)** : laissé au défaut `imageio-ffmpeg` (libx264 sur ffmpeg système). Si la taille des MP4 dépasse les 5 MB en T1.4 / T2.6, on bumpera via un kwarg `codec` mais pas avant.
- **`with` context manager** pour `save_video` : la fonction écrit en une passe, pas de ressource à libérer côté caller.
- **Logging des fichiers écrits** : volontairement silencieux ; l'evaluator (T0.12) trace déjà dans `result["videos"]`.

**Validation**
```bash
# Tests dédiés
.venv/bin/python -m unittest tests.test_video -v
# → Ran 9 tests, OK

.venv/bin/python -m unittest tests.test_plotting -v
# → Ran 7 tests, OK

# Suite complète
.venv/bin/python -m unittest discover tests/
# → Ran 124 tests, OK

# Smoke check vidéo (le critère d'acceptation PLAN.md)
.venv/bin/python -c "
import numpy as np
from src.utils.video import save_video
frames = [np.zeros((400, 600, 3), dtype=np.uint8)] * 30
out = save_video(frames, '/tmp/test_t013.mp4')
print('wrote', out, out.stat().st_size, 'bytes')
"
# → wrote /tmp/test_t013.mp4 <non-zero> bytes
```

---

### v0.7.0 — 2026-04-27 — Env factory + wrappers (T0.7)

**Type** : feature
**Ticket** : T0.7
**Phase** : 0 (Fondations)

**Résumé**
Ajout de `src/env/factory.py::build_env(cfg, seed, render_mode)` (point d'entrée unique pour instancier `LunarLander-v3`) et `src/env/wrappers.py` (registre `WRAPPERS` + wrappers custom `ObsNormalize`, `RewardClip` + alias `RecordEpisodeStats` vers `gym.wrappers.RecordEpisodeStatistics`). Couverture par 15 tests dans `tests/test_env_factory.py`.

**Contexte / motivation**
§4.2 de `PROJECT.md` impose une factory unique : « Aucun autre fichier que `factory.py` ne doit appeler `gym.make()` ». Le but est triple — (1) changer d'env = changer une clé YAML, (2) composer les wrappers de façon déclarative et persistable dans `runs/<run>/config.yaml`, (3) ajouter un wrapper bonus = ajouter une ligne au registre, sans toucher au trainer. Ce ticket débloque T0.11 (le trainer va appeler `build_env`), T0.12 (l'evaluator avec `render_mode="rgb_array"` pour la vidéo), Phase 1 (les baselines random/heuristic ont besoin de l'env), et tous les bonus environnementaux (vent T5.8, continuous T5.7).

**Détails techniques**

- **Fichiers créés** :
  - `src/env/factory.py` — fonction `build_env`.
  - `src/env/wrappers.py` — registre `WRAPPERS` + classes custom.
  - `tests/test_env_factory.py` — 15 tests.
- **Signature gelée** (cf. PLAN.md §Signatures) :
  ```python
  def build_env(cfg: EnvConfig, *, seed: int, render_mode: str | None = None) -> gym.Env
  ```
- **Pipeline factory** :
  1. `gym.make(cfg.id, render_mode=render_mode, **cfg.kwargs)` — `cfg.kwargs` permet de passer `enable_wind=True` ou `continuous=True` directement depuis le YAML.
  2. Compose les wrappers dans l'ordre de la liste YAML : `env = WRAPPERS[wrapper_cfg.name](env, **wrapper_cfg.kwargs)`. L'ordre compte (un `ObsNormalize` après `RewardClip` n'agit pas sur les mêmes flux).
  3. `env.reset(seed=seed)` puis `env.action_space.seed(seed)` — les deux seeds sont nécessaires : `reset(seed)` initialise la dynamique Box2D, `action_space.seed(seed)` rend `env.action_space.sample()` reproductible (utilisé par `RandomAgent` en T1.1).
  4. Wrapper inconnu → `ValueError` explicite avec liste des wrappers disponibles. Fail loud, pas de fallback silencieux.
- **Registre `WRAPPERS`** (`src/env/wrappers.py`) :
  ```python
  WRAPPERS = {
      "RecordEpisodeStats": gym.wrappers.RecordEpisodeStatistics,  # alias
      "ObsNormalize":       ObsNormalize,                          # custom
      "RewardClip":         RewardClip,                            # custom
  }
  ```
  - **`RecordEpisodeStats`** : alias vers `gym.wrappers.RecordEpisodeStatistics` (le nom officiel Gymnasium 1.0). Le YAML continue d'utiliser `RecordEpisodeStats` conformément aux exemples PROJECT.md §4.3 / PLAN.md, sans contredire le nom interne de la lib. Injecte `info["episode"] = {"r": return, "l": length, "t": wall_time}` au step terminal — sera consommé par le trainer T0.11 pour remplir les colonnes `return` / `length` du logger.
  - **`ObsNormalize(env, *, eps=1e-8)`** — z-score running stats avec algorithme de Welford (mean + M2 incrémentaux) :
    ```python
    delta = obs - mean
    mean += delta / count
    M2 += delta * (obs - mean)
    var = M2 / count
    return (obs - mean) / (sqrt(var) + eps)
    ```
    Output forcé en `np.float32` (cohérent avec `Transition.obs` PLAN.md §Signatures). Pas de dépendance à `gym.wrappers.NormalizeObservation` — implémentation locale 10 lignes, plus prévisible pour les tests et plus simple à débrancher en ablation.
  - **`RewardClip(env, *, low=-1.0, high=1.0)`** — sous-classe de `gym.RewardWrapper`, `reward(r) = np.clip(r, low, high)`. Validation `low <= high` au constructeur. Sera utilisé en ablation T3.4 (reward clipping ON/OFF).
- **Tests `tests/test_env_factory.py`** (15 cas, 4 classes) :
  - `TestBuildEnv` (8) : observation_space `(8,)`, action_space `Discrete(4)`, unwrapped est bien `LunarLander`, no-wrappers ne crée pas de couches parasites, ordre des wrappers respecté (`ObsNormalize(RewardClip(env))`), alias `RecordEpisodeStats` résolu, `ValueError` sur wrapper inconnu, seed déterministe (obs reset + 20 actions samplées identiques entre deux instances), `render_mode="rgb_array"` propagé jusqu'à l'env de base et `env.render()` renvoie un `(H, W, 3)` numpy.
  - `TestRewardClip` (4) : clip au-dessus, clip en-dessous, passthrough dans la plage, `low > high` lève `ValueError`.
  - `TestObsNormalize` (2) : shape `(8,)` + dtype `float32`, `count` augmente après quelques steps.
  - `TestRegistry` (1) : les 3 noms requis sont enregistrés.

**Alternatives rejetées**

- **Aliasser `ObsNormalize` vers `gym.wrappers.NormalizeObservation`** : la version Gymnasium normalise sur la base d'un `RunningMeanStd` interne avec une API légèrement différente (`update_running_mean`, `epsilon` par défaut différent). Rouler notre propre Welford donne contrôle total, zéro surprise au fil des versions Gymnasium, et 10 lignes lisibles dans le code source. Le surcoût est nul.
- **Aliasser `RewardClip` vers `gym.wrappers.ClipReward`** : `ClipReward` existe mais utilise `min_reward`/`max_reward` comme noms de kwargs alors que nos YAML utilisent `low`/`high`. Plutôt que de gérer le mapping de noms, on ré-implémente — c'est trivial.
- **Mettre la liste des wrappers dans une string CSV** plutôt qu'une `list[dict]` YAML : moins lisible et empêche les kwargs par wrapper. La structure `list[{name, kwargs}]` est imposée par `WrapperConfig` (T0.4).
- **Ne pas seeder `action_space`** : tentant car `env.reset(seed=seed)` est déjà fait, mais `action_space.sample()` a son propre RNG. Sans `action_space.seed(seed)`, `RandomAgent` produirait des actions différentes entre deux runs identiques — casserait `test_seeding.py::test_env_rollout_is_reproducible`.
- **Permettre `gym.make("LunarLander-v3")` ailleurs en dépannage** : violation directe de §4.2 de `PROJECT.md` (« everything env-related is built from config »). Pas de débat.
- **Cacher l'env construit via cache mémoïsé** (`@lru_cache` sur `(cfg, seed)`) : risque d'aliasing entre runs (un wrapper stateful comme `ObsNormalize` partagerait son `_count` entre seeds). Reconstruire à chaque appel = état frais garanti.
- **Faire une factory qui retourne aussi un `seed` dérivé** ou `env_info` dict : sur-conception, le contrat actuel est suffisant. Le trainer/evaluator récupèrent ce dont ils ont besoin via `env.observation_space` / `env.action_space` directement.
- **Utiliser un décorateur `@register_wrapper("name")`** comme prévu pour les agents : pour les wrappers le dict statique suffit (très peu d'entrées attendues). Le décorateur sera utile pour `agents/registry.py` (T0.8) où on s'attend à empiler beaucoup d'agents (random, heuristic, dqn, double_dqn, ppo, sac, ...).
- **Lever `ValueError` si `cfg.kwargs` contient une clé interdite** : la liste des kwargs valides dépend de l'env, on délègue cette validation à `gym.make` qui lève déjà une erreur claire.

**Impact**
- **T0.11 (Trainer)** : peut maintenant écrire `env = build_env(cfg.env, seed=cfg.seed)` et oublier complètement Gymnasium derrière. Le trainer reste env-agnostic.
- **T0.12 (Evaluator)** : pareil avec `render_mode="rgb_array"` quand `cfg.eval.record_video=True`.
- **T1.1 / T1.2 (baselines)** : `RandomAgent` peut sampler via `env.action_space.sample()` (deterministic seedé) ; `HeuristicAgent` peut lire les indices de l'obs vector (cf. §3.2 PROJECT.md) sans connaître l'env.
- **T3.4 (ablation reward clipping)** : il suffit d'ajouter `RewardClip` dans la liste de wrappers du YAML d'ablation, zéro changement de code.
- **T5.8 (bonus vent)** : YAML avec `env.kwargs.enable_wind: true, wind_power: 15.0, turbulence_power: 1.5` — passé tel quel à `gym.make`.
- **Aucune dette technique** : les wrappers custom sont auto-contenus (pas d'imports croisés vers `agents/`, `training/`, etc.). Le registre est extensible sans refactor.

**Non couvert volontairement**
- **Wrapper `RecordVideo`** : non listé par §4.1 de `PROJECT.md`. La vidéo sera gérée par `utils/video.py` (T0.13) qui consomme des frames `imageio` directement, plutôt que via un wrapper Gymnasium — donne plus de contrôle (FPS, taille, encodeur).
- **Wrapper `TimeLimit`** : déjà appliqué automatiquement par `gym.make` (Gymnasium garantit un `TimeLimit` par défaut). Pas besoin de l'ajouter explicitement.
- **`build_buffer(cfg)` / `build_network(cfg)` factories** : T0.9 / T0.10.
- **Validation de la config** (vérifier que `cfg.id` existe dans Gymnasium avant `gym.make`) : `gym.make` lève déjà une erreur claire. On ne double pas la validation.

**Validation**
```bash
# Tests dédiés (15 cas)
python -m unittest tests.test_env_factory -v
# → Ran 15 tests, OK

# Suite complète
python -m unittest discover tests/
# → Ran 55 tests, OK

# Smoke check manuel (factory + reset + step)
python -c "
from src.env.factory import build_env
from src.utils.config import EnvConfig, WrapperConfig
cfg = EnvConfig(id='LunarLander-v3', kwargs={}, wrappers=(
    WrapperConfig(name='RecordEpisodeStats', kwargs={}),
))
env = build_env(cfg, seed=0)
obs, _ = env.reset(seed=0)
assert obs.shape == (8,) and env.action_space.n == 4
print('OK', obs.dtype, obs.shape)
"
```

---

### v0.6.0 — 2026-04-27 — Classifieur de terminaison (T0.6)

**Type** : feature
**Ticket** : T0.6
**Phase** : 0 (Fondations)

**Résumé**
Ajout de `src/utils/termination.py` : enum `TerminationReason` (`LAND` / `CRASH` / `OUT_OF_VIEW` / `SLEEP` / `TRUNCATED` / `UNKNOWN`) et fonction `classify(info, reward, state, terminated, truncated)` qui mappe une fin d'épisode LunarLander-v3 sur l'une des catégories. Couverture par 18 tests synthétiques dans `tests/test_termination.py`.

**Contexte / motivation**
§3.5 et §7.2 de `PROJECT.md` exigent que chaque épisode soit loggué avec sa **raison** de terminaison (crash / land / out-of-view / sleep / truncated). C'est l'un des trois critères d'acceptation §7. Sans ce module, la colonne `termination_reason` de `metrics.csv` (cf. `Logger`, T0.5) ne pourrait être remplie. La logique est isolée dans un module dédié, testable hors environnement, pour pouvoir être exercée par 1000 trajectoires synthétiques sans dépendre de Box2D — la validation sur trajectoires réelles viendra en T1.3.

**Détails techniques**

- **Fichiers créés** :
  - `src/utils/termination.py` : enum + `classify()`.
  - `tests/test_termination.py` : 18 tests couvrant chaque branche + cas adverses.
- **Signature gelée** (cf. PLAN.md §Signatures) :
  ```python
  def classify(info: dict, reward: float, state: np.ndarray,
               terminated: bool, truncated: bool) -> TerminationReason
  ```
- **Heuristiques implémentées** (ordre de priorité) :
  1. **Override `info["termination_reason"]`** : si un wrapper a déjà classé l'épisode (valeur valide d'enum), on l'honore. Valeur invalide → fallback heuristique silencieux. Permet à un futur wrapper d'env d'injecter une vérité-terrain sans réécrire la fonction.
  2. **`truncated=True` → `TRUNCATED`** : prioritaire sur tout signal de récompense. Gymnasium peut renvoyer `terminated=True` ET `truncated=True` simultanément (rare mais possible) ; on traite la coupure max-steps comme dominante car elle reflète une décision externe à la dynamique.
  3. **`terminated=False` → `UNKNOWN`** : la fonction n'est pas censée être appelée hors fin d'épisode, mais on retourne `UNKNOWN` plutôt que de lever pour rester safe en logging.
  4. **Both legs contact + reward ≥ 80** → `LAND` : seuil à 80 (et non 100) pour absorber le coût per-step du dernier tick. `+100` est ajouté au moment du `not lander.awake` dans Gymnasium ; le step terminal contient aussi du shaping négatif (~-0.3 par thruster).
  5. **`|x| ≥ 1.0`** → `OUT_OF_VIEW` : limite de frame exacte de l'env (`abs(state[0]) >= 1.0` dans la source Gymnasium). Placée **avant** la branche `CRASH` car les deux donnent reward ≈ -100 ; la position permet seule de discriminer.
  6. **reward ≤ -80** → `CRASH` : touche le sol corps en premier, pas via les jambes. Seuil à -80 pour la même raison de symétrie que LAND.
  7. **No legs + |vx|, |vy|, |theta_dot| ≤ 0.05** → `SLEEP` : Box2D met le corps "asleep" lorsqu'il est immobile suffisamment longtemps. Cas distinct de LAND car aucun pied au contact (atterrissage borderline ou repos sur la coque sans crash franc).
  8. Sinon → `UNKNOWN`. Doit rester à 0 % sur trajectoires réelles ; sera vérifié en T1.3.
- **Constantes nommées** (`_LAND_REWARD_MIN=80.0`, `_CRASH_REWARD_MAX=-80.0`, `_OUT_OF_VIEW_X=1.0`, `_SLEEP_LIN_VELOCITY_MAX=0.05`, `_SLEEP_ANG_VELOCITY_MAX=0.05`, `_IDX_*`) — préfixe `_` pour souligner l'usage interne ; pas de surface publique au-delà de `classify` et `TerminationReason`.
- **Validation de forme** : `np.asarray(state, dtype=np.float32)` puis `shape == (8,)` requis. Rejet via `ValueError` explicite — fail loud plutôt que silently mis-classer.
- **Enum** : `class TerminationReason(str, Enum)` — héritage de `str` permet la comparaison directe `reason == "land"` et la sérialisation YAML/JSON gratuite (le logger sérialise `termination_reason` en colonne CSV).
- **Aucune dépendance ajoutée** : seulement `numpy` (déjà présent) et stdlib `enum` / `typing`. Pas d'appel à `gymnasium` — le module est testable en pure Python.

**Alternatives rejetées**

- **Lever `ValueError` quand `terminated=False and truncated=False`** : tentant pour signaler un usage incorrect, mais cela rendrait le logger plus fragile (un appel pendant un épisode en cours casserait tout le run). Retourner `UNKNOWN` est plus tolérant et le test T1.3 vérifiera de toute façon le taux de `UNKNOWN`.
- **Mettre la branche CRASH avant OUT_OF_VIEW** : ordre du code source de Gymnasium (`if game_over or abs(x)>=1`). Mais en l'absence d'accès à `game_over` (info dict ne le contient pas), on ne peut différencier que par la position, donc OOV doit être testé en premier sinon tout sortie hors-cadre serait classée CRASH par défaut.
- **Seuils stricts à exactement ±100** : trop fragiles. Le shaping per-step ajoute ~±0.3 au step terminal (penalty thruster, distance pad). Marges ±20 absorbent ces variations sans masquer les vrais signaux.
- **Détecter SLEEP via `info["TimeLimit.truncated"]` ou un flag Box2D** : `info` ne contient pas le flag `awake` du corps Box2D. Heuristique sur la vélocité est la seule observable disponible.
- **Heuristique "both legs + reward modéré → LAND souple"** (seuil à +50) : risque de fausser les statistiques (atterrissages "moyens" comptabilisés mais le sujet veut un vrai LAND avec +100 final). Le seuil à 80 reste robuste.
- **Mettre le module dans `src/training/` plutôt que `src/utils/`** : §4.1 de `PROJECT.md` impose `src/utils/termination.py`. Pas de débat.
- **Ajouter une constante `THRESHOLDS` exposée et configurable depuis YAML** : sur-conception. Les seuils sont fixes par la physique de l'env, pas un hyperparamètre d'entraînement. Si un futur env nécessitait d'autres bornes, on créera une variante.
- **Couvrir des cas partiels (un seul pied au contact + reward modéré)** : ces situations sont par construction `UNKNOWN`. Ne pas inventer une catégorie supplémentaire — laisser remonter au logger pour analyse en T1.3.

**Impact**
- **T0.5 (Logger)** : la colonne `termination_reason` peut maintenant être remplie par les valeurs `TerminationReason.X.value` (string lowercase).
- **T0.11 (Trainer)** : à chaque fin d'épisode, le trainer appellera `classify(info, reward, last_obs, terminated, truncated)` et passera la chaîne au logger.
- **T0.12 (Evaluator)** : agrège les `termination_reason` sur les épisodes d'éval pour produire la breakdown du `summary.json` (§8 de `PROJECT.md`).
- **T1.3** : valide expérimentalement que `UNKNOWN == 0` sur 100+100 épisodes random+heuristic — c'est l'épreuve du feu de ce classifieur.
- **Aucune dette technique** : les seuils sont localisés et le module est testé exhaustivement.

**Non couvert volontairement**
- **Validation sur trajectoires réelles** → T1.3 (nécessite RandomAgent/HeuristicAgent qui n'existent pas encore).
- **Hooking dans le trainer** → T0.11 (le trainer n'existe pas).
- **Documentation utilisateur** (README) → T4.1 / T6.2.
- **Adaptation au mode `continuous=True`** : la sémantique reste valide (mêmes 8D, mêmes terminations) mais non testée explicitement — viendra avec le bonus T5.7 (SAC continuous).

**Validation**
```bash
# Tests unitaires (18 cas dédiés)
python -m unittest tests.test_termination -v
# → Ran 18 tests, OK

# Suite complète
python -m unittest discover tests/
# → Ran 40 tests, OK
```
Run de référence : pas de run réel à ce stade — la validation finale (UNKNOWN ratio) est conditionnée à T1.3.

---

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
