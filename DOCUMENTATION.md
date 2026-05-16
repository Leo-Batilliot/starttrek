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

### v2.1.0 — 2026-05-16 — DQNAgent : squelette forward Q + ε-greedy (T2.1)

**Type** : feature
**Ticket** : T2.1
**Phase** : 2 (DQN MVP)

**Résumé**
Ajout de `src/agents/dqn.py::DQNAgent` (enregistré `"dqn"`) — **squelette** : construit `q_net` + `target_net` via `build_mlp`, sélectionne l'action en ε-greedy (ε = `training/schedules.linear_epsilon` du nombre de transitions observées), `observe()` pousse dans le `ReplayBuffer` et fait décroître ε. `learn()` est un no-op documenté (la cible bootstrap + le pas de gradient sont T2.2). Ajout de `src/training/schedules.py::linear_epsilon` (fonction pure, décroissance linéaire). Couverture : `tests/test_dqn_agent.py` (18) + `tests/test_schedules.py` (11). **Ouvre la Phase 2.**

**Contexte / motivation**
Premier ticket de la Phase 2 (§6.2 PROJECT.md). On découpe le DQN en deux : T2.1 = chemin avant + collecte (testable sans boucle d'apprentissage), T2.2 = `learn()` (bootstrap + Adam + sync target). Ce squelette est déjà branchable dans le trainer agent-agnostic existant (qui appelle `act`/`observe`/`learn` à chaque step) sans rien y modifier — un run "dqn" tournerait dès maintenant en collectant des transitions avec une politique non entraînée.

**Détails techniques**
- **Fichiers créés** :
  - `src/training/schedules.py` — `linear_epsilon(step, *, start, end, decay_steps) -> float`. Fonction **pure** (pas d'état, indépendante de l'agent/env → reproductible et testable isolément) : `step ≤ 0 → start` ; `step ≥ decay_steps → end` (clampé) ; sinon interpolation linéaire. `decay_steps ≤ 0 → ValueError`. Gère `start == end` (constant) et `start < end` (montant).
  - `src/agents/dqn.py` — `DQNAgent`. `tests/test_dqn_agent.py` (18 tests), `tests/test_schedules.py` (11 tests).
- **Fichiers modifiés** : `src/agents/__init__.py` (+`from . import dqn`, side-effect register) ; `PLAN.md` (case T2.1 cochée).
- **Constructeur** `DQNAgent(cfg, *, n_actions, obs_dim=8)` :
  - Le trainer et l'evaluator instancient `cls(cfg, n_actions=env.action_space.n)` (pas d'`obs_dim`). DQN ayant besoin de la dim d'entrée du MLP, `obs_dim` est un **kwarg optionnel** défaulté à `LUNARLANDER_OBS_DIM = 8` (constante module-level, §3.2 PROJECT.md / "Layout des données"), surchargeable pour les tests / autres envs discrets. Choix : pas de modification de la signature partagée trainer/eval (pas de breaking change), pas d'init paresseux (complexifie save/load).
  - `q_net` et `target_net` via `build_mlp(obs_dim, n_actions, list(cfg.network.hidden_sizes), cfg.network.activation)`. `target_net` = copie exacte (`load_state_dict`), `.eval()`, params `requires_grad_(False)` (gelée — sync en T2.2).
  - `ReplayBuffer(capacity=cfg.buffer_size, obs_shape=(obs_dim,))`.
  - RNG d'exploration : `np.random.default_rng(np.random.randint(...))` — même pattern d'isolement que `RandomAgent` (reproductible sous `seed_all`, découplé du RNG global).
  - `device = cpu` (cible MVP ≤ 2 h CPU, T2.4 ; déterminisme du seeding simplifié).
  - `self.epsilon` exposé (colonne `epsilon` stable de `metrics.csv`, lue par le trainer via `getattr`).
- **`act(obs, *, greedy=False) -> int`** : `_q_values` valide `obs.shape == (obs_dim,)` (fail loud, comme HeuristicAgent), `torch.no_grad()` forward, `argmax`. `greedy=True → argmax(Q)`. `greedy=False` : avec proba `ε(step)` action uniforme, sinon argmax — ε courant = `self.epsilon`.
- **`observe(transition)`** : `buffer.push` + incrémente `_step_count` + recalcule `self.epsilon = linear_epsilon(_step_count, …)`. ε décroît donc sur le **nombre de transitions vues** (le trainer appelle `observe` une fois par step env).
- **`save`/`load`** : `torch.save` d'un dict `{q_net, target_net, step_count, n_actions, obs_dim}` ; `load` valide `n_actions` **et** `obs_dim` (mismatch → `ValueError`, comme `RandomAgent`), restaure les state dicts + `_step_count` + ε. `weights_only=False` explicite (checkpoint local de confiance ; forward-compat torch ≥ 2.6).

**Alternatives rejetées**
- *Init paresseux des réseaux (inférer `obs_dim` au 1er `act`/`observe`)* : éviterait la constante 8 mais complique `save`/`load` (réseaux possiblement non construits) et l'invariant "instancié = prêt" du critère d'acceptation. Le kwarg défaulté est plus simple et explicite.
- *Modifier la signature trainer/eval pour passer `obs_dim=env.observation_space.shape[0]`* : breaking change sur des fichiers partagés, hors périmètre T2.1 (processus §2 : ne pas anticiper).
- *Implémenter `learn()` maintenant* : c'est explicitement T2.2 ; le no-op documenté garde le squelette minimal et le trainer fonctionnel (métriques vides gérées).
- *RNG d'exploration = `np.random` global* : moins isolé ; le `Generator` dédié reproduit le pattern éprouvé de `RandomAgent`.
- *`schedules` en classe à état* : une fonction pure est plus simple à tester (boundaries) et n'a pas besoin d'état (le compteur de pas vit dans l'agent).

**Impact**
- `AGENTS["dqn"]` instanciable et utilisable dans le trainer/eval existants sans modification ; collecte de transitions + politique greedy/ε-greedy opérationnelles.
- `linear_epsilon` réutilisable tel quel par d'éventuels autres agents ε-greedy (Double DQN bonus, etc.).
- Aucune perf/dette : pas d'optimiseur ni de boucle d'apprentissage tant que T2.2 n'est pas fait (volontaire).

**Non couvert volontairement**
- `learn()` : sample buffer + cible `y = r + γ·max Q_target·(1-done)` + Huber/MSE + Adam + grad clip + sync target → **T2.2**.
- `configs/baseline/dqn.yaml` → **T2.3** (le critère T2.1 instancie via un `AgentConfig` direct, pas un YAML).
- Run d'entraînement / jalon ≥ 200 → **T2.4**.

**Validation**
- Critère T2.1 : `AGENTS["dqn"](cfg, n_actions=4)` instancie sans erreur ; `act(obs, greedy=True) == argmax(Q(obs))` (vérifié contre `q_net` direct) ; `act(obs, greedy=False)` à ε=1.0 explore les 4 actions, à ε=0.05 (après `decay_steps` `observe`) → 193/200 greedy (mix selon ε). ✅
- `python -m unittest discover tests/` → **186 tests, OK** (1 skipped sans rapport ; +29 vs v1.4.0).

### v1.4.0 — 2026-05-16 — Assets vidéos des baselines (T1.4)

**Type** : infra (livrable assets — aucun code modifié)
**Ticket** : T1.4
**Phase** : 1 (Baselines)

**Résumé**
Extraction de deux vidéos MP4 propres pour le rapport — `assets/videos/random.mp4` (crash typique de `RandomAgent`) et `assets/videos/heuristic.mp4` (atterrissage propre de `HeuristicAgent`) — produites par `eval.py --record-video` et sélectionnées via une règle déterministe (premier épisode enregistré, `seed = cfg.seed = 0`). Premier contenu commité dans `assets/`. **Clôt la Phase 1.**

**Contexte / motivation**
Dernier ticket de la Phase 1 (§6.1 PROJECT.md). Les deux baselines (T1.1 random, T1.2 heuristic) et la validation du classifier (T1.3) sont closes ; il reste à figer le matériel visuel "ce que fait un agent non-appris" qui servira de point de comparaison dans le rapport et la soutenance face au DQN entraîné (T2.6 / showcase T4.3). Aucune logique à écrire : le pipeline d'enregistrement existe déjà (`evaluator._save_video` → `utils/video.save_video`, v0.13.0) ; T1.4 est une étape de **curation reproductible**.

**Détails techniques**
- **Fichiers ajoutés** :
  - `assets/videos/random.mp4` — 15 979 B, 1,10 s, H.264 (avc1) yuv420p 600×400 @ 60 fps. Copie de `runs/random/0/videos/episode_0000.mp4`. Épisode 0 : `return = -48.07`, 65 steps, terminaison `crash` (cohérent avec le breakdown random 91 % crash / 9 % out_of_view).
  - `assets/videos/heuristic.mp4` — 29 391 B, 2,55 s, H.264 (avc1) yuv420p 600×400 @ 60 fps. Copie de `runs/heuristic/0/videos/episode_0000.mp4`. Épisode 0 : `return = 297.35`, 152 steps, terminaison `land` (atterrissage textbook ; breakdown heuristic 84 % land).
- **Fichiers modifiés** : `PLAN.md` — case T1.4 cochée.
- **Règle de sélection (déterministe, reproductible)** : `video_every = 25` sur 100 épisodes d'éval → épisodes 0, 25, 50, 75 enregistrés. On retient **`episode_0000.mp4`** : c'est le premier épisode produit par `eval.py`, dont le reset env utilise `seed = cfg.seed + 0 = 0` (cf. `evaluator.evaluate`, `env.reset(seed=seed + ep_idx)`). Choix canonique, stable, trivialement rejouable — pas un "meilleur épisode" cherry-pické.
- **Encodage** : inchangé par rapport à la sortie d'`eval.py` — `imageio.mimwrite(..., fps=60, macro_block_size=1)` (libx264 via `imageio-ffmpeg`). Pas de ré-encodage ni de trim : les fichiers sont déjà ~3 ordres de grandeur sous le plafond de 5 MB.
- **Détail repo** : `assets/{plots,videos}/.gitkeep` sont ignorés par la règle `.*` de `.gitignore` (jamais commités) ; `assets/` n'avait donc aucun fichier tracké jusqu'ici — ces deux MP4 sont son premier contenu versionné. Vérifié : `git check-ignore` ne matche pas les `.mp4`.

**Alternatives rejetées**
- *Script d'automatisation sélection+copie (`scripts/make_baseline_videos.py`)* : anticiperait `reproduce.sh` (T3.5) ; le critère d'acceptation ancre explicitement la repro sur la commande `eval.py` existante, pas sur un nouvel outil. Une règle de sélection documentée suffit et reste plus simple (processus §2 : ne pas anticiper un ticket ultérieur).
- *Choisir l'épisode au meilleur return (heuristic ep75 = 300.39)* : non déterministe vis-à-vis d'un changement de `video_every` ou de seed, et "cherry-pick" non défendable en soutenance. "Premier épisode enregistré" est une règle invariante.
- *Ré-encoder / compresser (crf, downscale, trim ≤ 30 s)* : inutile (15,6 KB / 28,7 KB ≪ 5 MB ; 1,1 s / 2,55 s ≪ 30 s) et introduirait une dépendance ffmpeg CLI explicite hors du pipeline `imageio` du projet.
- *Montage multi-épisodes en une vidéo* : le ticket demande "une vidéo propre de chaque baseline", pas un montage ; un épisode unique représentatif est plus lisible pour le rapport.

**Impact**
- Matériel visuel des baselines figé et versionné → réutilisable tel quel dans le rapport (T4.2) et la soutenance, base de comparaison avant/après apprentissage.
- Confirme que la chaîne `eval.py --record-video` → `assets/` est reproductible bit-pour-bit (env seedé, encodage déterministe).

**Non couvert volontairement**
- Vidéo du DQN entraîné → **T2.6** (`assets/videos/dqn_trained.mp4`).
- Vidéo showcase 20-30 s avec incrustation du return → **T4.3** (`assets/videos/showcase.mp4`).
- Plots de training → **T2.5**.

**Validation**
- Repro : `python eval.py --run runs/random/0 --record-video` → `mean_return = -185.70`, breakdown `crash 0.91 / out_of_view 0.09` ; `python eval.py --run runs/heuristic/0 --record-video` → `mean_return = 252.83`, breakdown `land 0.84` (identiques à v1.3.0 — déterminisme confirmé). Génère `runs/<baseline>/0/videos/episode_0000.mp4`.
- Critères T1.4 : `assets/videos/random.mp4` (15 979 B, 1,10 s) et `assets/videos/heuristic.mp4` (29 391 B, 2,55 s) — **≤ 5 MB** et **≤ 30 s** chacune ; décodage `ffmpeg -v error -f null -` sans erreur (visionnables, H.264 standard).
- `python -m unittest discover tests/` → 157 tests, `OK` (aucun code modifié — inchangé vs v1.3.0).

### v1.3.0 — 2026-05-15 — Validation de `termination.classify` sur trajectoires réelles (T1.3)

**Type** : fix (+ test, + tooling)
**Ticket** : T1.3
**Phase** : 1 (Baselines)

**Résumé**
Correction de `src/utils/termination.py::classify` pour s'appuyer sur le **contrat de reward terminal** réel de LunarLander-v3 (±100 fixés par l'env), ce qui élimine les `UNKNOWN` sur trajectoires réelles. Ajout du script de validation dédié `scripts/validate_termination.py` qui rejoue les deux baselines (RandomAgent + HeuristicAgent) sur 100 épisodes greedy chacune et **asserte le critère T1.3** (`UNKNOWN == 0` + breakdown plausible), avec log écrit dans `events.log`. Run de référence : random → **0 % unknown** (90 % crash / 10 % out_of_view), heuristic → **0 % unknown** (84 % land / 11 % sleep / 2 % crash / 2 % out_of_view / 1 % truncated), `mean_return` heuristic = 252.83 (identique à v1.2.0, reproductible). Le **26 % `unknown`** laissé ouvert par T1.2 est résorbé.

**Contexte / motivation**
T1.2 avait délibérément exposé un bug : 26 % des fins d'épisodes de l'heuristique tombaient en `UNKNOWN`. Cause racine : l'ancien `classify` exigeait **les deux jambes** en contact pour `LAND` et détectait `SLEEP` par seuils de vitesse — deux heuristiques fragiles. Or LunarLander-v3 (`gymnasium/envs/box2d/lunar_lander.py`, `step()`, lignes 656-662) **écrase** le reward du pas terminal à une sentinelle fixe : `-100` (crash corps-sol *ou* `|x| >= 1.0` hors champ) ou `+100` (corps Box2D endormi = au repos). La truncation (wrapper `TimeLimit`) garde le reward shapé et met `truncated=True`. Classer à partir du **signe de cette sentinelle** rend `UNKNOWN` structurellement impossible sur trajectoires réelles, ce qui est exactement le critère d'acceptation T1.3.

**Détails techniques**
- **Fichiers créés** :
  - `scripts/validate_termination.py` — entrypoint de validation (réutilise `load_config` / `build_env` / `AGENTS` / `evaluate` / `Logger`, aucune logique dupliquée). Sortie : `runs/validate_termination/{events.log,metrics.csv}` + stdout, exit code `0`/`84`.
- **Fichiers modifiés** :
  - `src/utils/termination.py` — `classify` réécrit autour du contrat de reward terminal. Nouvelle logique : `truncated → TRUNCATED` ; `not terminated → UNKNOWN` ; `reward ≥ +80 → LAND` si une jambe en contact sinon `SLEEP` ; `reward ≤ -80 → OUT_OF_VIEW` si `|x| ≥ 1.0` sinon `CRASH` ; garde défensive `UNKNOWN` pour input non-stock. Constantes mortes supprimées (seuils de vitesse `SLEEP`), `_TERMINAL_REWARD_MAGNITUDE` introduit. **Signature inchangée** → `trainer.py` et `evaluator.py` non impactés.
  - `tests/test_termination.py` — 2 tests qui encodaient l'ancienne sémantique physiquement fausse réécrits : `test_land_requires_both_legs` → `test_land_with_single_leg_contact` (une jambe + sentinelle +100 = `LAND`) ; `test_sleep_at_rest_without_leg_contact` → `test_sleep_when_asleep_without_leg_contact` (sentinelle +100 sans jambe = `SLEEP`) ; `test_sleep_not_triggered_when_velocity_high` → `test_mid_range_terminal_reward_is_unknown_guard`. Les 14 autres tests passent inchangés.
  - `PLAN.md` — case T1.3 cochée.
- **Décision notable** : la sentinelle de reward est la source de vérité (déterministe, documentée dans le code source de l'env) ; les booléens `leg1/leg2` ne servent plus qu'à départager `LAND` vs `SLEEP` à reward +100, et `|x|` à départager `CRASH` vs `OUT_OF_VIEW` à reward -100.

**Alternatives rejetées**
- *Garder l'ancienne heuristique et baisser les seuils* : ne corrige pas la cause (un atterrissage où une seule jambe est flaggée à l'obs finale reste mal classé) ; reporte le bug.
- *Mapper `UNKNOWN` arbitrairement vers `LAND`/`SLEEP`* : ferait passer le critère sans le comprendre ; non défendable en soutenance.
- *Assertion runtime dans `evaluator.py`* (proposée comme option dans le ticket) : coupling de la boucle d'éval à une contrainte de validation + édition de code partagé ; le script dédié est plus propre et exécutable à la demande / dans `reproduce.sh`.
- *Test unitaire sur trajectoires réelles* : lancer 200 épisodes Box2D dans `unittest` serait lent et fragile vis-à-vis de l'OS ; le contrat est verrouillé par les tests synthétiques rapides, la validation réelle par le script.

**Impact**
- `UNKNOWN` structurellement impossible sur LunarLander-v3 réel → `termination_breakdown` de tous les runs (random, heuristic, futur DQN) est fiable pour les plots et le `summary.json` (§8 PROJECT.md).
- `scripts/validate_termination.py` réutilisable tel quel dans `reproduce.sh` (T3.5).
- Distinction `LAND`/`SLEEP` : un repos sans contact jambe est compté `SLEEP` (11 % chez l'heuristique) — fidèle à §3.5 PROJECT.md.

**Non couvert volontairement**
- Extraction des vidéos baselines propres → **T1.4** (`assets/videos/random.mp4`, `heuristic.mp4`).
- Validation sur trajectoires DQN entraînées → implicite en Phase 2 (le contrat ne dépend pas de la politique).

**Validation**
- `python scripts/validate_termination.py --episodes 100` → `OVERALL: PASS`, `events.log` contient `unknown_count=0` + `verdict=PASS` pour random, heuristic et `ALL`.
- Valeurs attendues : random `unknown=0.0` (crash+out_of_view = 1.00) ; heuristic `unknown=0.0`, `land=0.84`, `mean_return=252.83`.
- `python -m unittest discover tests/` → 157 tests, `OK` (1 skipped, sans rapport).

### v1.2.0 — 2026-05-12 — HeuristicAgent (T1.2)

**Type** : feature
**Ticket** : T1.2
**Phase** : 1 (Baselines)

**Résumé**
Ajout de `src/agents/heuristic_agent.py::HeuristicAgent` (contrôleur PD-style sur l'obs 8-dim de LunarLander-v3, sans état, sans apprentissage) enregistré sous `"heuristic"` dans le registry. Auto-import dans `src/agents/__init__.py`. Création de `configs/baseline/heuristic.yaml` (20 000 steps, eval 100 épisodes). Couverture par 14 tests (TestHeuristicAgent). Run de référence `runs/heuristic/0/` : **mean_return = 252.83** sur 100 épisodes greedy (CI95 [237.38, 268.29]), termination_breakdown **58 % land** / 2 % crash / 2 % out_of_view / 11 % sleep / 1 % truncated / **26 % unknown** — largement au-dessus du critère T1.2 (`mean_return ≥ 0` + proportion `LAND` visible). Le 26 % `unknown` est attendu et constitue précisément le matériel d'entrée de T1.3 (validation de `termination.classify` sur trajectoires réelles).

**Contexte / motivation**
Deuxième baseline obligatoire (§6.1 PROJECT.md). Trois fonctions :
1. **Borne haute "sans apprentissage"** — une politique closed-form bien calibrée doit poser le lander régulièrement. Si plus tard le DQN ne dépasse pas l'heuristique, on aura mal réglé l'algorithme : l'heuristique fixe le niveau "raisonnable" que tout agent appris devrait au moins égaler. Ici le DQN visera ≥ 200 ; on a déjà 253 avec une formule fermée — l'écart entre heuristique et DQN sera surtout un gain en robustesse/régularité, pas en plafond.
2. **Stress-test de `termination.classify` sur trajectoires "qui se terminent par un atterrissage"** — RandomAgent ne produisait quasi aucun `LAND` (91 % crash, 9 % out_of_view, 0 % land). L'heuristique en produit 58 %, ce qui exerce enfin la branche `LAND` du classifier sur des trajectoires réelles. Et révèle (volontairement) le bug ouvert : 26 % de fins d'épisodes sont classées `UNKNOWN`, ce qui aurait été invisible avec RandomAgent seul.
3. **Vidéo "ce qu'un bon agent fait"** — `runs/heuristic/0/videos/` contient 4 MP4 d'atterrissages réussis, utiles comme matériel visuel pour le rapport et la soutenance.

**Détails techniques**

- **Fichiers créés** :
  - `src/agents/heuristic_agent.py` — `HeuristicAgent`.
  - `configs/baseline/heuristic.yaml` — config baseline.
  - `tests/test_heuristic_agent.py` — 14 tests.
- **Fichiers modifiés** :
  - `src/agents/__init__.py` — ajout de `from . import heuristic_agent` (side-effect register).
- **Politique `HeuristicAgent._heuristic(obs)`** (contrôleur PD à 2 boucles imbriquées) :
  - **Décomposition de l'obs** : `[x, y, v_x, v_y, theta, theta_dot, leg1, leg2]` (cf. §3.2 PROJECT.md).
  - **Boucle d'orientation** :
    - `angle_targ = clip(0.5*x + 1.0*v_x, ±0.4)` — on penche le lander dans le sens opposé à la dérive horizontale (proportionnel à x et v_x). Le clip à ±0.4 rad (~23°) évite les pencher excessifs qui font perdre le contrôle.
    - `angle_todo = 0.5*(angle_targ - theta) - 1.0*theta_dot` — terme P sur l'erreur d'angle + terme D sur la vitesse angulaire (anti-oscillation).
  - **Boucle de hauteur** :
    - `hover_targ = 0.55*|x|` — altitude de "hover" qui augmente quand on est loin du centre. Idée : si on est loin, on garde de l'altitude pour avoir le temps de manœuvrer ; si on est centré, on peut descendre.
    - `hover_todo = 0.5*(hover_targ - y) - 0.5*v_y` — PD sur l'altitude.
  - **Override sur contact** : `leg1 or leg2 → angle_todo = 0, hover_todo = -0.5*v_y`. Une fois posé, on ne corrige plus l'angle (les pattes amortissent) et on freine la vitesse verticale restante.
  - **Décision** :
    - `hover_todo > |angle_todo| and hover_todo > 0.05` → `2` (main engine, priorité à la chute si urgente).
    - sinon `angle_todo < -0.05` → `3` (right thruster), `> 0.05` → `1` (left thruster).
    - sinon `0` (do nothing).
  - **Coefficients exposés comme constantes module-level** (`ANGLE_TARG_GAIN_X`, `ANGLE_KP`, etc.) pour permettre à un futur ticket d'ablation de les surcharger sans toucher au corps de la fonction. Pas exposés dans `AgentConfig` — l'heuristique est une référence, pas un objet à tuner.
- **`HeuristicAgent(cfg, *, n_actions)`** :
  - `n_actions != 4 → ValueError`. L'heuristique encode hard-coded les ids d'action {0, 1, 2, 3} de LunarLander discret ; un env continu (`continuous=True`, 2D continu) ou un env à action_space différent n'a pas de mapping cohérent.
  - `act(obs)` : valide `obs.shape == (8,)` (fail loud), retourne `int` Python (cohérent avec `RandomAgent` et `NoOpAgent`, le trainer passe tel quel à `env.step`).
  - `observe`/`learn` : no-ops (`learn() → {}`).
  - `save` : crée un fichier vide (idem `NoOpAgent`) pour satisfaire le pipeline de checkpoints. Aucun état à sérialiser : la politique est stateless, ses coefficients sont des constantes module-level.
  - `load` : no-op.
  - `epsilon = 0.0` exposé pour le logger (RandomAgent expose `1.0`, NoOpAgent `0.0`, DQN exposera son ε courant — colonne `epsilon` stable dans `metrics.csv`).
  - **`act(greedy)` ignore le flag** : l'heuristique est purement déterministe. `act(obs)` et `act(obs, greedy=True)` retournent strictement la même action — testé.
- **`configs/baseline/heuristic.yaml`** :
  - Calque structurel sur `configs/baseline/random.yaml` (mêmes clés, même schéma `AgentConfig`).
  - `epsilon.start = end = 0.0` (sémantique : 0 % exploration — politique déterministe).
  - Le reste (`network`, `gamma`, `lr`, `buffer_size`, `target_update`, `grad_clip`) renseigné à des valeurs neutres car `AgentConfig` exige tous les champs (cf. v0.14.0). Aucun de ces champs n'est lu par `HeuristicAgent`.
  - `training.total_steps: 20000` → 78 épisodes en pratique (épisodes longs car le lander vole/hover plus longtemps avant terminaison vs RandomAgent qui crashe vite). Suffisant pour le mean_return rapporté.
  - `eval.episodes: 100` (critère §7.1 PROJECT.md), `video_every: 25` (4 vidéos sur 100 épisodes).
- **Tests `test_heuristic_agent.py` (14)** :
  - `test_registered_under_heuristic` : `AGENTS["heuristic"] is HeuristicAgent`.
  - `test_is_an_agent` : instance valide de l'ABC.
  - `test_invalid_n_actions_raises` : `n_actions ∈ {0, 2} → ValueError` (4 obligatoire).
  - `test_act_returns_int_in_range` : 200 obs aléatoires → tous `int ∈ [0, 4)`.
  - `test_act_is_deterministic` : 100 obs → `act(o)` == `act(o)` == `act(o, greedy=True)` (politique sans état + sans aléa).
  - `test_act_rejects_wrong_shape` : `obs.shape == (7,) → ValueError`.
  - `test_falling_too_fast_fires_main` : obs(y=0, vy=-1) → action 2 (chute rapide à l'altitude cible → main engine).
  - `test_drifting_right/left_fires_correcting_thruster` : obs(x=±0.5, vx=±0.2) → action 1/3 (correction d'orientation).
  - `test_stable_centered_does_nothing` : obs nulle → action 0.
  - `test_legs_contact_neutralizes_angle` : `legs=1` avec orientation pourrie → action 0 (override de l'override : on ne corrige plus l'angle quand posé).
  - `test_observe_and_learn_are_safe` : pas de crash, `learn() == {}`.
  - `test_save_load_roundtrip` : save → load → politique inchangée (stateless).
  - `test_exposes_epsilon_attribute` : `agent.epsilon == 0.0`.

**Alternatives rejetées**

- **Coefficients dans `AgentConfig.heuristic`** : aurait nécessité d'étendre le dataclass typé avec un sous-champ optionnel. Inutilement intrusif — l'heuristique est une référence figée, pas un agent à tuner. Si plus tard on veut faire une ablation sur les gains, on rajoutera une sous-classe `TunedHeuristicAgent` plutôt que de polluer `AgentConfig`.
- **Implémenter une variante stochastique pour `act(greedy=False)`** (ex. bruit gaussien sur les seuils) : double l'effort de test, ajoute du bruit non motivé. L'heuristique est censée être un **niveau de référence reproductible** ; l'aléa appartient à RandomAgent.
- **Reposer sur `env.action_space.sample()` ou tout autre fallback** : l'heuristique doit produire une décision sur **toutes** les observations, sinon elle perd son intérêt d'oracle. Pas de "je sais pas → random".
- **Coefficients alternatifs** (ex. plus agressifs sur l'angle) : la combinaison choisie est le PD bien connu de l'exemple Gymnasium `lunar_lander.heuristic`, déjà éprouvée. Pas de raison de re-tuner sans grille systématique, et un tuning systématique est hors scope (§2.1 PROJECT.md interdit le tuning automatique).
- **Tolérer `n_actions != 4` en mode "best effort"** (ex. clip de l'action à `n_actions-1`) : faux silence. Mieux vaut un `ValueError` clair que de retourner des actions absurdes en cas de mauvaise config.
- **Conserver le `unknown` 26 % sans note** : non, il est explicitement signalé dans cette entrée comme **input attendu de T1.3** (qui doit ramener UNKNOWN à 0). Le ticket T1.2 a un critère "proportion LAND visible" (58 %, validé), pas "UNKNOWN == 0" (porté par T1.3).

**Impact**
- **T1.3 (validation termination réelle)** débloqué — on a maintenant les 100 épisodes heuristic exigés par le ticket, qui montrent un bug observable du classifier (26 % UNKNOWN). T1.3 va consister à inspecter ces cas dans `metrics.csv` et raffiner `classify(info, reward, state, terminated, truncated)`.
- **T1.4 (vidéos baselines)** débloqué — `runs/heuristic/0/videos/episode_*.mp4` candidates pour `assets/videos/heuristic.mp4`.
- **Phase 2 (DQN)** : on a maintenant **deux** bornes — -185 (random) et +253 (heuristic). Un DQN qui atterrit entre les deux est mal réglé. Le critère DQN ≥ 200 est en dessous de l'heuristique en plafond, mais l'enjeu est la régularité (CI95) et la robustesse au vent (T5.8), pas le plafond brut.
- **Dette technique** : aucune. HeuristicAgent est self-contained (~85 lignes), stateless, sans dépendance hors numpy.

**Non couvert volontairement**
- **Validation `UNKNOWN == 0` comme assertion runtime** → T1.3.
- **Export propre des vidéos baselines vers `assets/videos/`** → T1.4.
- **Multi-seed runs** → l'heuristique est déterministe et l'env LunarLander a un aléa de spawn ; faire seeds 0..4 pour heuristique n'apporterait que la variance de spawn, qui est déjà capturée par les 100 épisodes du single run. Pas pertinent (contrairement au DQN où la variance d'init des poids justifie 5 seeds).
- **Tuning des coefficients PD** : hors sujet — l'heuristique est une référence, pas un agent à optimiser.

**Validation**
```bash
# Tests dédiés
.venv/bin/python -m unittest tests.test_heuristic_agent -v
# → Ran 14 tests, OK

# Suite complète (skip attendu : test_random_not_registered_yet)
.venv/bin/python -m unittest discover tests/
# → Ran 157 tests, OK (skipped=1)

# Critère d'acceptation T1.2
.venv/bin/python train.py --config configs/baseline/heuristic.yaml --seed 0
# → 78 episodes, mean_return_last_100 = 259.55, wall_time 1.1s

.venv/bin/python eval.py --run runs/heuristic/0 --episodes 100 --greedy --record-video
# → mean_return = 252.83, std = 78.86, CI95 [237.38, 268.29]
# → solved (≥ 200 over 100) : True
# → termination_breakdown: 58% land, 2% crash, 2% out_of_view, 11% sleep, 1% truncated, 26% unknown
# → 4 videos in runs/heuristic/0/videos/
```
Run de référence : `runs/heuristic/0/`.

---

### v1.1.0 — 2026-05-11 — RandomAgent (T1.1)

**Type** : feature
**Ticket** : T1.1
**Phase** : 1 (Baselines)

**Résumé**
Ajout de `src/agents/random_agent.py::RandomAgent` (échantillonnage uniforme sur `[0, n_actions)` via un `np.random.Generator` privé) enregistré sous `"random"` dans le registry. Auto-import dans `src/agents/__init__.py`. Création de `configs/baseline/random.yaml` (20 000 steps, eval 100 épisodes). Couverture par 10 tests (TestRandomAgent). Run de référence `runs/random/0/` : **mean_return = -185.70** sur 100 épisodes greedy, breakdown 91 % crash / 9 % out_of_view / 0 % land — cohérent avec le bruit attendu d'une politique aléatoire sur LunarLander-v3. Premier livrable de Phase 1.

**Contexte / motivation**
Premier agent réel du projet et baseline obligatoire (§6.1 PROJECT.md). Sert trois fonctions :
1. **Borne inférieure de performance** — fournit le `mean_return` à battre pour qu'un agent appris ait un sens (un DQN qui plafonne au niveau random est cassé).
2. **Validation de la chaîne sur trajectoires réelles** — le NoOp produisait des fins quasi-identiques (90+ steps de chute libre → crash) ; RandomAgent introduit de la variance (out_of_view apparaît, longueurs d'épisode plus dispersées), exerçant `termination.classify` sur des cas non triviaux. C'est T1.3 qui validera formellement `UNKNOWN == 0`, mais déjà ici 0/100 unknowns.
3. **Premier vrai usage du pipeline vidéo** — `eval.py --record-video` produit 4 MP4 (`video_every: 25` sur 100 épisodes) qui serviront de référence visuelle dans le rapport (T1.4 fera l'export propre vers `assets/videos/random.mp4`).

**Détails techniques**

- **Fichiers créés** :
  - `src/agents/random_agent.py` — `RandomAgent`.
  - `configs/baseline/random.yaml` — config baseline.
  - `tests/test_random_agent.py` — 10 tests.
- **Fichiers modifiés** :
  - `src/agents/__init__.py` — ajout de `from . import random_agent` (side-effect register).
- **`RandomAgent(cfg: AgentConfig, *, n_actions: int)`** :
  - **Constructor** : dérive son propre seed via `int(np.random.randint(0, 2**31 - 1))` au moment de l'instanciation. Comme `train.py` appelle `seed_all(seed)` *avant* de construire l'agent, ce tirage est déterministe pour un seed global donné → reproductibilité bout-à-bout sans avoir à propager `cfg.seed` jusqu'à l'agent (le `RunConfig` ne l'expose pas dans `AgentConfig`).
  - **RNG privé** : `self._rng = np.random.default_rng(self._seed)`. Pas d'usage de `np.random.*` global pour éviter de polluer l'état partagé (le trainer pourrait à terme se servir lui aussi de `np.random` pour d'autres choses, et on veut que les actions de l'agent restent isolées et persistables).
  - **`act(obs, *, greedy=False) → int`** : `self._rng.integers(0, n_actions)`. Le flag `greedy` est ignoré (une politique aléatoire n'a pas de "greedy" — c'est documenté dans le `assert` du test).
  - **`observe`/`learn`** : no-op (`learn()` renvoie `{}`, géré gracieusement par le trainer via `if "loss" in metrics`).
  - **`save(path)`** : pickle `{seed, bit_generator_state, n_actions}` (cf. PLAN.md "save/load = dump/load d'un seed"). `bit_generator_state` est un dict NumPy déjà sérialisable, contient toute la position courante du PRNG — un `load` suivi de `act()` reprend EXACTEMENT là où le `save` s'est arrêté.
  - **`load(path)`** : check `n_actions` vs checkpoint pour éviter qu'un eval avec un env différent (ex. `continuous=True` à 2D) charge un checkpoint discret incompatible. Lève `ValueError` clair.
  - **Attribut `self.epsilon: float = 1.0`** : RandomAgent = 100 % exploration permanente. Permet au logger de remplir la colonne `epsilon` sans `None`.
- **`configs/baseline/random.yaml`** :
  - `run_name: random`, `seed: 0`, env `LunarLander-v3` avec wrapper `RecordEpisodeStats`.
  - Tous les hyperparams `agent.network/gamma/lr/buffer_size/...` renseignés à des valeurs réalistes même si RandomAgent les ignore — `AgentConfig` exige tous les champs (volontaire, cf. v0.14.0). `epsilon.start = end = 1.0` pour cohérence sémantique ("exploration full").
  - `training.total_steps: 20000` → ~200 épisodes (assez pour une moyenne stable, < 1 s de wall-time).
  - `eval.episodes: 100` (matche le critère §7.1 PROJECT.md), `video_every: 25` (4 vidéos par eval, suffisant pour le rapport sans gonfler le dossier).
- **Tests `test_random_agent.py` (10)** :
  - `test_registered_under_random` : `AGENTS["random"] is RandomAgent`.
  - `test_is_an_agent` : instance valide de l'ABC.
  - `test_act_returns_in_range` : 200 appels, tous `int` ∈ `[0, n)`.
  - `test_act_explores_all_actions` : sur 2000 tirages avec `n=4`, chaque action est sortie > 200 fois (test statistique loose : E = 500, marge x2.5).
  - `test_reproducible_with_seed_all` : `seed_all(42)` → `RandomAgent()` → 50 actions ; idem deux fois → mêmes séquences.
  - `test_different_seeds_differ` : `seed_all(0)` vs `seed_all(1)` → séquences distinctes.
  - `test_observe_and_learn_are_safe` : pas de crash, `learn() == {}`.
  - `test_save_load_roundtrip` : 17 actions, save, 30 actions captées comme "futur attendu", création d'un nouvel agent avec seed différent, load, 30 actions → identiques aux 30 captées. Vérifie que `bit_generator.state` capture bien la position du PRNG.
  - `test_load_rejects_n_actions_mismatch` : save avec n=4, load dans agent n=2 → `ValueError`.
  - `test_invalid_n_actions_raises` : `n_actions=0 → ValueError`.

**Alternatives rejetées**

- **`act` via `env.action_space.sample()` (déjà seeded par `build_env`)** : violerait l'isolation agent/env (`Agent.act(obs)` ne reçoit pas l'env). Aussi, `env.action_space.sample()` est seeded *une fois* à `build_env`, partagé avec d'autres callers potentiels — moins propre que notre RNG privé.
- **`act` via `np.random.randint` global** : marche pour la reproductibilité mais corromprait l'état partagé NumPy (si le trainer / un wrapper l'utilise ailleurs). Le `Generator` privé est l'approche moderne recommandée par NumPy ≥ 1.17.
- **Renvoyer `int(np.random.randint(...))` dans `act` plutôt que `Generator.integers`** : `Generator.integers` est la nouvelle API stable ; le `RandomState` legacy est marqué deprecated dans la doc NumPy. Pas de raison de coder en legacy.
- **`save` au format JSON** : `bit_generator.state` est un dict imbriqué (`{"bit_generator": "PCG64", "state": {"state": <bignum>, "inc": <bignum>}, "has_uint32": ..., "uinteger": ...}`) avec des entiers Python potentiellement très grands. JSON serialise ça mais pickle est strict-equal sans gymnastique. Le `.pt` n'est pas torch-spécifique ici, juste l'extension convention du projet.
- **`save` via `torch.save`** pour cohérence d'extension avec DQN : tirerait `torch` dans `random_agent.py` pour rien. Pickle stdlib suffit, RandomAgent reste libre de toute dep RL/ML.
- **`act(greedy=True)` retourne `0`** : faux sens. "Greedy" suppose qu'il existe une distribution de préférence à maximiser ; pour un random uniform il n'y en a pas. Garder le tirage uniforme rend explicite que l'agent est intrinsèquement stochastique.
- **`total_steps: 5000`** comme smoke : trop court pour avoir un `mean_return` stable (20k → 218 épisodes, std ~108, CI95 ±21 ; à 5k on n'aurait que ~55 épisodes, CI95 ±~30).
- **`record_video: true` dans le YAML** : non — `eval.py` consomme cette clé seulement si on n'overrride pas via `--record-video`. Garder `false` par défaut évite de générer des vidéos systématiquement à chaque eval (utile pour CI / boucles seed). T1.4 fera l'export propre via la flag CLI.
- **Mettre RandomAgent dans un fichier `baselines.py` partagé avec HeuristicAgent** : couplage gratuit. Le projet impose un fichier par agent (`agents/random_agent.py`, `agents/heuristic_agent.py`) pour que le registre reste découpé et que T5.X (Double DQN, PPO, SAC) ajoutent simplement un fichier de plus.
- **Skip de `test_random_not_registered_yet` est un signal négatif** : non — c'est exactement le contrat documenté dans `test_base_agents.py::TestRegistry::test_random_not_registered_yet` : `self.skipTest("random agent already registered (T1.1 done)")`. Le test reste utile en CI tant que T1.1 n'est pas livré ; une fois ce ticket clos, il documente proprement le ticket franchi.

**Impact**
- **Phase 1 démarrée** : T1.2 (HeuristicAgent) suit le même pattern (un fichier d'agent + un YAML baseline + un test) sans toucher au trainer/eval/factory.
- **T1.3 (validation termination réelle)** : `runs/random/0/eval_*.json` montre 0 UNKNOWN sur 100 épisodes, premier datapoint sur 200 (100 random + 100 heuristic) attendus par le ticket.
- **T1.4 (vidéos baselines)** : `runs/random/0/videos/episode_0000.mp4` (≈ 25 KB) candidate pour `assets/videos/random.mp4` — à vérifier qualité visuelle puis copie.
- **Phase 2 (DQN)** : RandomAgent fournit la baseline numérique. Quand T2.4 dira `mean_return ≥ 200`, on aura l'amplitude du gain (`200 - (-185) = +385` points).
- **Dette technique** : aucune. RandomAgent est self-contained (40 lignes), n'introduit aucune dépendance, et est facilement détachable.

**Non couvert volontairement**
- **`HeuristicAgent`** → T1.2.
- **Validation `UNKNOWN == 0`** comme assertion formelle → T1.3 (le run montre 0 mais ce n'est pas encore un test runtime).
- **Export propre des vidéos baselines vers `assets/videos/`** → T1.4.
- **Plots agrégés multi-seed** (un seul seed run ici) → T3.1 / T3.6 (pas pertinent pour RandomAgent, mais utile pour DQN).
- **CLI `--record-video` exposé dans `train.py`** : non, c'est le rôle d'`eval.py` (T0.12). Les vidéos pendant le training n'ont pas de sens (politique encore en cours d'évolution).

**Validation**
```bash
# Tests dédiés
.venv/bin/python -m unittest tests.test_random_agent -v
# → Ran 10 tests, OK

# Suite complète (skip attendu : test_random_not_registered_yet)
.venv/bin/python -m unittest discover tests/
# → Ran 143 tests, OK (skipped=1)

# Critère d'acceptation T1.1
.venv/bin/python train.py --config configs/baseline/random.yaml --seed 0
# → 218 episodes, mean_return_last_100 = -185.19, wall_time 1.0s

.venv/bin/python eval.py --run runs/random/0 --episodes 100 --greedy --record-video
# → mean_return = -185.70, std = 108.16, CI95 [-206.90, -164.51]
# → termination_breakdown: 91% crash, 9% out_of_view
# → 4 videos in runs/random/0/videos/
```
Run de référence : `runs/random/0/`.

---

### v0.14.0 — 2026-05-11 — Smoke config + run end-to-end (T0.14)

**Type** : feature
**Ticket** : T0.14
**Phase** : 0 (Fondations)

**Résumé**
Ajout de `src/agents/noop_agent.py::NoOpAgent` (placeholder zéro-action enregistré sous `"noop"` dans `AGENTS`) ; auto-enregistrement via side-effect import dans `src/agents/__init__.py` ; mise à jour de `configs/smoke.yaml` pour pointer sur l'agent `noop` (au lieu de `random` qui sera fourni par T1.1) ; nettoyage d'un import parasite (`from tests.test_logger import METRIC_COLUMNS`) dans `train.py`. La commande `python train.py --config configs/smoke.yaml --seed 0` boucle 5000 steps sans crash et produit le dossier `runs/smoke/0/` complet (`config.yaml`, `metrics.csv`, `events.log`, `checkpoints/{step_*.pt, final.pt}`). Couverture par 7 tests `NoOpAgent` + 2 tests smoke end-to-end (suite totale **133 / 133 OK**).

**Contexte / motivation**
Dernier ticket de la Phase 0. Le but n'est pas algorithmique — c'est de **valider la chaîne complète** (config YAML → seed → factory env → registry agent → trainer → logger → checkpoints) en mettant tout bout-à-bout. Sans agent réel (T1.1 fournira `RandomAgent`), il fallait un placeholder qui (a) satisfait l'ABC `Agent`, (b) ne perturbe pas les futurs tests "T1.1 pas encore fait" (`test_random_not_registered_yet`), (c) ne traîne pas de dette technique une fois T1.1 livré. La fin de Phase 0 ouvre Phase 1 (baselines) sans dépendance restante côté infra.

**Détails techniques**

- **Fichiers créés** :
  - `src/agents/noop_agent.py` — `NoOpAgent`.
  - `tests/test_noop_agent.py` — 7 tests.
  - `tests/test_smoke.py` — 2 tests (chargement config + run trainer end-to-end raccourci).
- **Fichiers modifiés** :
  - `src/agents/__init__.py` — `from . import noop_agent  # noqa: F401` (side-effect register).
  - `configs/smoke.yaml` — `agent.name: noop` (au lieu de `random`, qui sera réintroduit dans `configs/baseline/random.yaml` à T1.1).
  - `train.py` — suppression de `from tests.test_logger import METRIC_COLUMNS` (import depuis le dossier `tests/` dans du code prod = anti-pattern, et celui-ci shadowait silencieusement l'import canonique depuis `src.training.trainer`).
- **`NoOpAgent(cfg: AgentConfig, *, n_actions: int)`** :
  - Signature constructor identique à celle attendue par `train.py::main` (`agent_class(cfg.agent, n_actions=env.action_space.n)`) — préserve l'invariant "tout agent du registry est instanciable par le trainer sans cast spécial".
  - `act(obs, *, greedy=False) → 0` : action 0 = "do nothing" sur LunarLander-v3 (cf. §3.2 PROJECT.md). Sortie Python `int` → le trainer la passe telle quelle à `env.step` (Gymnasium accepte `int` pour `Discrete`).
  - `observe`, `learn`, `load` : no-ops. `learn()` retourne `{}` (le trainer gère déjà l'absence de clé `"loss"` via `if "loss" in metrics`).
  - `save(path)` : crée un fichier vide (et `mkdir parents=True`) pour que les checkpoints existent quand le trainer fait `save_every` + `final.pt`. Permet à l'evaluator (`eval.py::_resolve_checkpoint`) de trouver `final.pt` sans erreur.
  - Attribut `self.epsilon: float = 0.0` exposé pour que le trainer puisse logger la colonne `epsilon` via `getattr(agent, "epsilon", None)` sans forcer un `None` partout (les schedules viendront en T2.1).
  - Validation : `n_actions <= 0 → ValueError` (fail loud, cohérent avec les autres constructeurs du projet).
- **Auto-enregistrement** :
  - `src/agents/__init__.py` importe `noop_agent` au chargement du package. Quand `train.py` fait `from src.agents.registry import AGENTS`, Python initialise d'abord `src.agents` (donc `__init__.py` → `noop_agent` → décorateur `@register_agent("noop")` → entrée dans `AGENTS`), puis charge `registry`. Au moment où `train.py` interroge `AGENTS["noop"]`, l'entrée est là.
  - Pas de circularité : `noop_agent.py` n'importe que `src.agents.base` (terminal), `src.agents.registry` (terminal), et `src.utils.config` (terminal). Aucun de ces modules ne rappelle vers `src.agents.__init__`.
  - Vérifié : `test_random_not_registered_yet` (TestRegistry) reste vert — il check `"random"`, pas `"noop"`.
- **`configs/smoke.yaml`** — pointe désormais sur `agent.name: noop`. Les autres champs (`gamma`, `lr`, `batch_size`, etc.) restent renseignés à des valeurs réalistes pour qu'un `RunConfig` valide soit chargé même si `NoOpAgent` les ignore — utile pour ne pas dupliquer la config quand T1.1 réutilisera ce squelette.
- **Tests `test_noop_agent.py` (7)** :
  - `test_registered_under_noop` : `AGENTS["noop"]` est bien `NoOpAgent`.
  - `test_is_an_agent` : instance valide de l'ABC `Agent`.
  - `test_act_returns_zero` : `act(obs) == 0` et `act(obs, greedy=True) == 0`.
  - `test_observe_and_learn_are_safe` : `observe(Transition)` ne lève rien, `learn() == {}`.
  - `test_save_creates_file_and_load_no_op` : fichier créé même dans un dossier inexistant (mkdir parents) ; `load` ne lève rien.
  - `test_invalid_n_actions_raises` : `n_actions=0 → ValueError`.
  - `test_exposes_epsilon_attribute` : l'attribut `epsilon` existe pour le logger.
- **Tests `test_smoke.py` (2)** :
  - `test_smoke_config_loads` : `load_config("configs/smoke.yaml")` renvoie un `RunConfig` cohérent.
  - `test_trainer_runs_end_to_end_short` : version raccourcie (400 steps, save_every=200) du smoke pour rester rapide en CI. Construit env via factory, instancie NoOp, fait tourner `Trainer.run()`, puis vérifie : `summary["episodes"] > 0`, présence de `config.yaml`/`metrics.csv`/`events.log`, `checkpoints/final.pt` créé, header `metrics.csv` = `METRIC_COLUMNS`, chaque `termination_reason` est dans l'enum.
- **Validation manuelle** (commande exacte du critère d'acceptation) :
  ```bash
  python train.py --config configs/smoke.yaml --seed 0 --out-dir /tmp/smoke_runs
  ```
  Résultat observé : 5000 steps, 70 épisodes, wall_time 0.2 s, dossier `runs/smoke/0/` complet (config.yaml 627 B, metrics.csv 3.5 KB, events.log 12 KB, 6 checkpoints).

**Alternatives rejetées**

- **Anticiper T1.1 et coder directement `RandomAgent`** : casserait l'ordre des tickets. `RandomAgent` est livré par T1.1 avec son propre config baseline (`configs/baseline/random.yaml`) et son ticket de validation (T1.4 produit `assets/videos/random.mp4`). Mélanger T0.14 + T1.1 brouillerait la trace dans `DOCUMENTATION.md` et compliquerait la rétrospective.
- **Enregistrer `NoOpAgent` sous le nom `"random"`** : tentant pour ne pas toucher `smoke.yaml`, mais ça casserait `test_random_not_registered_yet` qui sert de contrat "T1.1 pas encore livré". Faux gain.
- **Ne pas auto-importer `noop_agent` dans `__init__.py`** et exiger `import src.agents.noop_agent` explicitement dans `train.py` : marche mais reproduit le travail à chaque nouvel agent (T1.1 random, T1.2 heuristic, T2.1 dqn, …). Centraliser dans `__init__.py` une fois et oublier. Les autres tickets ajouteront leur ligne dans le même fichier.
- **Importer dynamiquement via `importlib` + walk du dossier `src/agents/`** : sur-conception. Une liste statique d'imports est lisible, traçable par `git blame`, et donne une erreur claire si un module manque.
- **Faire `NoOpAgent.save(path)` un vrai no-op (pas de fichier)** : casserait l'invariant "post-training il existe un `final.pt`" attendu par `eval.py::_resolve_checkpoint` (Phase 1 utilisera `eval.py` sur ce run smoke). Un fichier vide est gratuit et marque proprement l'événement.
- **Action retournée par `act` = `n_actions // 2`** ou aléatoire : "no-op" implique zéro intent. Action 0 sur LunarLander = "do nothing engines off", la sémantique la plus proche d'un agent inactif. Aléatoire = pseudo-RandomAgent (T1.1).
- **Garder l'import `from tests.test_logger import METRIC_COLUMNS`** : code prod qui importe depuis `tests/` est un anti-pattern (couplage inversé, risque qu'un changement de tests casse la prod). Les deux constantes ont la même valeur — la version canonique est dans `src.training.trainer` (déjà importée à la ligne au-dessus). Le doublon était simplement un copier-coller à nettoyer.
- **Faire tourner le test smoke sur 5000 steps** (au lieu de 400) : trop lent en CI (~0.2 s ici mais l'env LunarLander a une variance forte selon la machine). Le critère d'acceptation 5000-step est validé manuellement et documenté ; le test unitaire en exécute 400 pour rester < 1 s.
- **Skip du smoke test si Box2D pas dispo** : le projet pin `gymnasium[box2d]` dans `requirements.txt` (cf. v0.1.0). On assume l'environnement est correctement installé. Pas de fallback silencieux.

**Impact**
- **Phase 0 close** : la chaîne CLI tourne end-to-end avec un agent factice ; les 9 critères de fin de Phase 0 (cf. PLAN.md §"Fin de Phase 0") sont remplis.
- **T1.1 (RandomAgent)** : démarrera sur un terrain solide — il suffira d'ajouter `src/agents/random_agent.py` + `from . import random_agent` dans `src/agents/__init__.py` + `configs/baseline/random.yaml`. Aucun changement de signature ni de trainer.
- **T1.2 (HeuristicAgent)** : même pattern.
- **T2.1 (DQN)** : idem, et héritera de la chaîne de checkpoints/eval déjà rodée par ce smoke.
- **CI / dev loop** : la 134-test suite (incluant le smoke unitaire) tourne en ~4 s — feedback rapide.
- **Dette technique introduite** : aucune. `NoOpAgent` est self-contained, supprimable trivialement quand il deviendra obsolète (ou conservable comme outil de diagnostic infra — il sert de smoke check permanent).

**Non couvert volontairement**
- **`RandomAgent` (uniform over `action_space`)** → T1.1.
- **`HeuristicAgent` (PD-style controller)** → T1.2.
- **Validation de `termination.classify` sur trajectoires réelles** → T1.3 (le smoke produit déjà du `crash`/`out_of_view` réel sur `metrics.csv` ; T1.3 vérifiera formellement `UNKNOWN == 0`).
- **Eval pendant training** : le trainer a un `if global_step % eval_every == 0: pass` — c'est volontaire (Phase 0 ne fait pas d'eval périodique en cours de training, T2.x le branchera quand un agent réel apprend).
- **Suppression de `NoOpAgent` après T1.1** : non, on le garde — utile pour vérifier que la chaîne reste intacte au fil des refactors, et coûte 30 lignes.
- **`summary.json` écrit sur disque** : `Trainer.run()` renvoie un dict mais ne le persiste pas (T2.4 le fera quand il y aura quelque chose d'intéressant à mesurer — `mean_return_last_100`, `solved`, etc.).

**Validation**
```bash
# Tests dédiés
.venv/bin/python -m unittest tests.test_noop_agent -v
# → Ran 7 tests, OK

.venv/bin/python -m unittest tests.test_smoke -v
# → Ran 2 tests, OK

# Suite complète
.venv/bin/python -m unittest discover tests/
# → Ran 133 tests, OK

# Critère d'acceptation T0.14 (commande exacte du PLAN.md)
.venv/bin/python train.py --config configs/smoke.yaml --seed 0 --out-dir /tmp/smoke_runs
# → 5000 steps, 70 episodes, wall_time 0.2s
# → /tmp/smoke_runs/smoke/0/{config.yaml, metrics.csv, events.log, checkpoints/{step_1000..5000.pt, final.pt}}
```

---

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
- **Implémentation** (corrigée en v1.3.0 — T1.3) : `utils/termination.py::classify(info, reward, state, terminated, truncated) → TerminationReason` (enum), basée sur le **contrat de reward terminal** de LunarLander-v3 (l'env écrase le reward du pas terminal à ±100) :
  - `truncated=True` → `TRUNCATED` ; `not terminated` → `UNKNOWN`.
  - `terminated=True` + `reward ≥ +80` (sentinelle +100, corps endormi) → `LAND` si une jambe en contact, sinon `SLEEP`.
  - `terminated=True` + `reward ≤ -80` (sentinelle -100) → `OUT_OF_VIEW` si `|x| ≥ 1.0`, sinon `CRASH`.
  - Garde défensive `UNKNOWN` pour input non-stock uniquement — **0 sur trajectoires réelles**, asserté par `scripts/validate_termination.py` (100 random + 100 heuristic).
- **Tests** : `tests/test_termination.py` couvre chaque cas avec des `info`/`reward`/`state` synthétiques ; la validation sur trajectoires réelles est faite par `scripts/validate_termination.py` (T1.3).

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
