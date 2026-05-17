# Apprendre à poser un module lunaire par Deep Q-Learning

**Projet Start Trek — Epitech AIA-401** · Gabin Wilson · promo 2026 · 2026-05-17
Environnement : Gymnasium `LunarLander-v3` (Box2D, discret) · Code : PyTorch + Gymnasium uniquement

---

## Résumé

Nous entraînons un agent **Deep Q-Network (DQN)** écrit à la main à résoudre
`LunarLander-v3`. Le critère de réussite du sujet est un **retour épisodique
moyen ≥ 200** sur **100 épisodes greedy** consécutifs. Sur **5 seeds (0–4)**,
la configuration de référence atteint **267.21** de retour moyen agrégé
(IC95 [256.19, 278.22]), avec **5/5 seeds résolus**. Nous mettons en évidence
et quantifions une **instabilité tardive du DQN vanilla, dépendante du seed**
(politique finale ≈ 212 vs ≈ 267 au pic, un seed sur cinq sous le seuil),
gérée par un mécanisme de *best-checkpoint*. Deux ablations contrôlées (un seul paramètre
modifié, vérifié par diff de configuration) sont menées : (A) la **taille du
replay buffer** (100 k → 10 k) produit une dégradation **statistiquement
significative** (IC95 disjoints, ≈ −9 %) ; (B) la **profondeur du réseau**
(2 → 3 couches cachées) produit un gain **non significatif** (IC95 recouvrants,
+2.7 %). Un **2ᵉ agent (Double DQN)** confirme le choix du DQN vanilla :
performance de pic équivalente (266.77, **p=0.95**) pour un code plus simple.
Tous les chiffres sont traçables vers des runs committés et intégralement
reproductibles via `reproduce.sh`.

---

## 1. Introduction et problème

`LunarLander-v3` est un problème de contrôle épisodique. À chaque pas, l'agent
observe un vecteur **s ∈ ℝ⁸** — position (x, y), vitesse (vₓ, v_y), angle θ,
vitesse angulaire ω, et deux booléens de contact des jambes — et choisit une
action discrète **a ∈ {0,1,2,3}** (ne rien faire, moteur gauche, moteur
principal, moteur droit). Le reward dense récompense le rapprochement de la
plateforme, pénalise l'usage des moteurs, et écrase le pas terminal à **+100**
(atterrissage) ou **−100** (crash / sortie de cadre). Un épisode se termine par
atterrissage, crash, sortie de vue, ou troncature (timeout / corps endormi).

L'objectif, fixé par le sujet, est un **mean episodic
return ≥ 200** sur **100 épisodes consécutifs en politique greedy, sans vent**,
reporté avec sa distribution et sa variance, sur les seeds reproductibles 0–4.
Ce rapport présente la méthode, le résultat de référence, l'analyse des modes
de terminaison, et deux ablations contrôlées avec lecture statistique.

## 2. Algorithme

Nous implémentons le **DQN** de Mnih et al. (2015) sans extension. L'agent
maintient un réseau de valeur Q_φ(s, a) et un **réseau cible** Q_φ⁻ figé. Les
transitions (s, a, r, s′, done) sont stockées dans un **replay buffer uniforme**
(ring-buffer NumPy) et échantillonnées par mini-batch. La cible de bootstrap
d'un échantillon est :

  **y = r + γ · maxₐ′ Q_φ⁻(s′, a′) · (1 − done)**

et le réseau est mis à jour par descente de gradient sur la perte de Huber
entre Q_φ(s, a) et y, avec **gradient clipping**. L'exploration suit une
politique **ε-greedy** dont ε décroît linéairement de 1.0 à 0.05. Le réseau
cible est synchronisé par **copie dure** tous les N pas. Ces choix (buffer
uniforme, MLP, ε linéaire, target hard) constituent volontairement le DQN
*vanilla* : les extensions (Double DQN, Dueling, PER, n-step…) sont des bonus
hors périmètre du MVP.

## 3. Dispositif expérimental

**Réseau.** MLP `8 → 128 → 128 → 4`, activation ReLU, initialisation He.
Nombre de paramètres ≈ **18.2 k** (référence) ; l'ablation B porte ce réseau à
`8 → 128 → 128 → 128 → 4`, ≈ **34.7 k** paramètres.

**Hyperparamètres (référence, `configs/baseline/dqn.yaml`).**

| Paramètre | Valeur | Paramètre | Valeur |
|---|---|---|---|
| γ (discount) | 0.99 | buffer size | 100 000 |
| learning rate | 5 × 10⁻⁴ | min buffer | 1 000 |
| batch size | 64 | target update | hard, /1000 pas |
| ε start → end | 1.0 → 0.05 | ε decay | 100 000 pas |
| grad clip | 10.0 | total steps | 500 000 |
| optimiseur | Adam | éval | 100 ép. greedy |

**Protocole d'évaluation.** Tous les 10 000 pas, l'agent est évalué (20
épisodes greedy). En fin de run, le **meilleur** checkpoint et le checkpoint
final sont évalués sur **100 épisodes greedy** ; le retour reporté est
`mean_return_last_100` (cf. §6). Chaque configuration est répétée sur **5 seeds
(0–4)**.

**Reproductibilité.** `seed_all()` seede Python, NumPy, Torch (CPU/CUDA) et
`env.reset(seed=…)` ; `torch.use_deterministic_algorithms(True)` est activé
quand faisable. La config exacte de chaque run est re-sérialisée dans
`runs/<run>/<seed>/config.yaml`. Le script `reproduce.sh` régénère les 15 runs
(3 configs × 5 seeds), les agrégats et toutes les figures.

**Agrégation et statistique.** Pour n = 5 seeds, nous reportons la moyenne et
un **intervalle de confiance à 95 %** sous approximation normale
(z = 1.96, écart-type d'échantillon ddof = 1). Le choix de z (plutôt que la loi
de Student, t₄ = 2.776) est un compromis de cohérence interne, assumé : avec
n = 5 les IC sont indicatifs, et nous fondons les conclusions de
**signification** sur le **recouvrement ou la disjonction** des IC, lecture
robuste à ce choix. Matériel : CPU 20 cœurs ; ≈ 30–47 min de wall-time par run.

**Classification des terminaisons.** `termination.classify()` distingue
`land / crash / out_of_view / sleep / truncated` à partir du contrat de reward
terminal (±100) et de l'état (contacts jambes, |x|). Validé à **0 %
d'`unknown`** sur 100 épisodes random + 100 heuristic (T1.3).

## 4. Baselines de référence

Deux politiques non apprenantes bornent le problème (100 épisodes greedy) :

| Baseline | Retour moyen | σ | Résolu | Terminaisons dominantes |
|---|---|---|---|---|
| Aléatoire | −185.70 | 108.2 | non | crash 0.91, out_of_view 0.09 |
| Heuristique (PD) | 252.83 | 78.9 | oui | land 0.84, sleep 0.11 |

L'agent aléatoire crashe quasi systématiquement. Le contrôleur heuristique
codé à la main franchit déjà le seuil de 200 — il fixe une cible exigeante :
le DQN doit non seulement résoudre la tâche mais **égaler ou battre** une
politique experte écrite à la main.

## 5. Résultat principal — DQN de référence

Sur 5 seeds, configuration de référence, best-checkpoint, 100 épisodes greedy :

| Seed | Retour moyen | σ | Résolu | first_solve_step |
|---|---|---|---|---|
| 0 | 259.53 | 48.0 | oui | 110 000 |
| 1 | 274.63 | 28.6 | oui | 140 000 |
| 2 | 273.00 | 32.4 | oui | 90 000 |
| 3 | 279.75 | 27.1 | oui | 110 000 |
| 4 | 249.13 | 39.9 | oui | 210 000 |
| **Agrégé** | **267.21** | — | **5/5** | **132 000** |

**Retour moyen agrégé = 267.21, IC95 [256.19, 278.22], 5/5 seeds résolus.**
Le critère du sujet (≥ 200) est **atteint et dépassé de ≈ 34 %**, et de façon
robuste : le minimum inter-seed (249.13) reste largement au-dessus du seuil.
Le pas de première résolution moyen est de 132 000 (IC95 [90.7 k, 173.3 k]),
soit ≈ 26 % du budget d'entraînement — la majeure partie des pas sert à
consolider, pas à atteindre le seuil.

![Courbe de retour agrégée (5 seeds, moyenne ± IC95) — référence DQN.](assets/plots/agg_returns_baseline.png)

## 6. Le phénomène d'effondrement tardif

Une observation scientifique importante : le **DQN vanilla est instable en fin
d'entraînement**, de façon **dépendante du seed**. Évaluée sur 100 épisodes
greedy, la politique **finale** (`final.pt`, step 500 k) rend en moyenne
**≈ 212** contre **≈ 267** pour le best-checkpoint : un écart d'environ
**−55 points**, concentré sur certains seeds — la majorité reste correcte
(232 / 225 / 268 / 280) mais un seed s'effondre nettement (**≈ 54**). C'est le
**deadly triad** classique (approximation de fonction + bootstrapping +
off-policy) : sans correctif, les valeurs Q peuvent diverger après avoir
convergé — pas systématiquement, mais de façon imprévisible selon le seed.

Conséquence méthodologique : nous conservons un **best-checkpoint** (meilleure
éval périodique) et reportons ses performances, en plus du checkpoint final.
Sans ce mécanisme, la politique finale serait **non fiable** (un seed sur cinq
sous le seuil) — distinction essentielle, souvent passée sous silence. Le
soft-target (Polyak) a été **testé** (5 seeds, `configs/bonus/dqn_softtarget.
yaml`) : équivalent au hard sur le pic (268.51 vs 267.21, Welch p=0.87) mais
il **n'élimine pas** cette instabilité finale (moyenne `final.pt` ≈ 160, un
seed à ≈ −119) — le best-checkpoint reste donc nécessaire.

## 7. Analyse des modes de terminaison

Répartition des terminaisons sur les évaluations greedy (best-checkpoint) :

| Configuration | land | crash | sleep | truncated |
|---|---|---|---|---|
| DQN référence (typ.) | 0.90–0.98 | 0.00–0.02 | 0.00–0.02 | 0.02–0.07 |
| Heuristique | 0.84 | 0.02 | 0.11 | 0.01 |
| Aléatoire | 0.00 | 0.91 | 0.00 | 0.00 |

Le DQN appris **atterrit proprement dans 90–98 % des cas**, contre 0 % pour
l'aléatoire (qui crashe) et 84 % pour l'heuristique (qui « s'endort » plus
souvent en stationnant sans se poser). La quasi-absence de `crash` et
`out_of_view` confirme une politique sûre. La fraction de `truncated`
(stationnement prolongé) est le principal mode d'échec résiduel du DQN —
piste d'amélioration (reward shaping anti-hover, ou pénalité de temps).

## 8. Ablation A — Taille du replay buffer

**Hypothèse.** Un buffer plus petit ⇒ échantillons plus corrélés et oubli plus
rapide ⇒ apprentissage plus instable et plafond plus bas.
**Protocole.** Un seul paramètre modifié : `buffer_size : 100000 → 10000`
(diff de config vérifié programmatiquement) ; 5 seeds.

| Métrique (agrégé 5 seeds) | Référence (100 k) | Ablation (10 k) |
|---|---|---|
| Retour moyen, IC95 | **267.21** [256.19, 278.22] | **242.40** [236.08, 248.72] |
| Résolu | 5/5 | 5/5 |
| σ par seed (plage) | 27–48 | 66–97 |
| first_solve_step moyen | 132 000 | 94 000 |

**Lecture.** Les IC95 sont **disjoints** (borne basse référence 256.19 > borne
haute ablation 248.72) : la dégradation de **≈ −25 points (−9 %)** est
**statistiquement significative**. Le petit buffer reste suffisant pour
*résoudre* (5/5 ≥ 200), mais avec une **variance inter-épisode 2–3× plus
élevée** (oubli catastrophique plus marqué). Contre-intuitivement, il atteint
le seuil **plus tôt** (94 k vs 132 k) : un buffer plus petit reflète une
politique plus récente (moins de retard de distribution), accélérant la montée
initiale au prix d'un plafond plus bas et plus bruité. **Compromis net :
buffer large = démarrage plus lent, mais plateau plus haut et plus stable.**

![Courbes appariées — ablation taille du replay buffer (10 k vs 100 k).](assets/plots/paired_buffer10k_vs_baseline.png)

## 9. Ablation B — Profondeur du réseau

**Hypothèse.** Une couche cachée supplémentaire (plus de capacité)
améliore-t-elle la performance asymptotique, ou le MLP 2 couches sature-t-il
déjà la tâche ?
**Protocole.** Un seul paramètre modifié : `hidden_sizes : [128,128] →
[128,128,128]` (≈ 18.2 k → 34.7 k paramètres) ; 5 seeds.

| Métrique (agrégé 5 seeds) | Référence (2×128) | Ablation (3×128) |
|---|---|---|
| Retour moyen, IC95 | **267.21** [256.19, 278.22] | **274.30** [265.25, 283.36] |
| Résolu | 5/5 | 5/5 |
| Retour min / max | 249.13 / 279.75 | 256.36 / 282.57 |
| first_solve_step moyen | 132 000 | 122 000 |

**Lecture.** Les IC95 se **recouvrent largement** : le gain moyen
(**+7.1 points, +2.7 %**) est **non significatif**. La 3ᵉ couche **double les
paramètres sans bénéfice mesurable** : sur cette tâche, le MLP 2×128 a déjà
une capacité suffisante. Résultat *négatif* utile, à mettre en regard de
l'ablation A : **toutes les ablations ne se valent pas — la donnée (taille du
buffer) pèse ici nettement plus que la capacité (profondeur)**. Les deux
configurations présentent le même effondrement tardif (§6).

![Courbes appariées — ablation profondeur du réseau (3×128 vs 2×128).](assets/plots/paired_depth3_vs_baseline.png)

## 10. Bonus — Double DQN (2ᵉ agent, justification du choix)

Pour **justifier le choix du DQN vanilla** comme MVP, nous comparons à un
second algorithme apprenant : **Double DQN** (van Hasselt et al., 2016). Le
`max` de la cible vanilla sélectionne *et* évalue l'action next avec le même
réseau cible → **surestimation systématique de Q**. Double DQN découple :
l'action next est choisie par le réseau **online**, évaluée par le réseau
**cible** : `y = r + γ·Q_cible(s', argmax Q_online(s', ·))`. **Seul ce calcul
change** (override d'une méthode ; aucune autre modification du code —
principe « un agent = un fichier »).

| Métrique (agrégé 5 seeds) | DQN référence | Double DQN |
|---|---|---|
| Retour moyen, IC95 | **267.21** [256.19, 278.22] | **266.77** [258.64, 274.91] |
| Résolu | 5/5 | 5/5 |
| `first_solve_step` moyen | 132 000 | 168 000 |
| Wall-time / run | ~46 min | ~22 min |

**Lecture.** Différence de performance asymptotique **nulle et non
significative** : Δ = −0.43 (−0.2 %), IC95 **recouvrants**, Welch
t=0.06, **p=0.95**, Cohen d=0.04. Double DQN **n'améliore pas le pic** sur
LunarLander (le best-checkpoint capte déjà le meilleur de chaque run), mais
(i) son IC95 est **plus resserré** (largeur 16 vs 22 → légèrement plus
régulier d'un seed à l'autre), (ii) il **réduit la surestimation de Q** par
construction (vérifié : cible Double ≤ `max` vanilla terme à terme), (iii) il
**entraîne ~2× plus vite** en wall-time. Conclusion : sur ce problème, le
**DQN vanilla est le bon choix MVP** (performance de pic équivalente, code
plus simple) ; Double DQN reste pertinent sous **budget CPU serré** (~2×
plus rapide) et comme **piste** contre l'instabilité finale (§6) — hypothèse
non encore validée sur `final.pt`, le soft-target seul n'ayant pas suffi.

![Courbes appariées — DQN vs Double DQN (moyenne ± IC95, 5 seeds).](assets/plots/paired_double_dqn_vs_baseline.png)

## 11. Recommandations

Recommandations actionnables, chacune tracée aux chiffres ci-dessus :

- **Livrer la baseline DQN vanilla** (`configs/baseline/dqn.yaml`, 2×128,
  buffer 100 k) : meilleur pic (267.21, 5/5), code minimal. Double DQN est
  équivalent (p=0.95) mais pas supérieur → préférer le plus simple.
- **Garde-fou critique — toujours évaluer `best.pt`, jamais `final.pt`.**
  La politique finale se dégrade selon le seed (§6 : moyenne ≈ 212 vs ≈ 267,
  un seed sur cinq sous le seuil). Le best-checkpoint est indispensable.
- **Ne pas réduire le replay buffer sous ~100 k.** Le passer à 10 k coûte
  **−9 % significatif** (p=0.008, IC disjoints) et **triple la variance**.
  Sous contrainte mémoire seulement, accepter ce compromis (résout encore
  5/5 mais instable, démarre plus tôt).
- **Garder le réseau à 2×128.** Une 3ᵉ couche n'apporte rien de significatif
  (p=0.36) pour le double de paramètres et de compute.
- **Sous budget temps serré, utiliser Double DQN** : même performance, wall
  ~2× plus court.
- **Stabilité finale : encore ouverte.** Le soft-target (Polyak) a été
  testé et **ne suffit pas** seul (§6). Piste restante : soft **+**
  Double DQN combinés, à valider sur `final.pt` ; en attendant, le
  best-checkpoint reste le garde-fou.

## 12. Discussion et limites

- **n = 5 seeds** : suffisant pour le critère du sujet et pour des IC
  indicatifs ; signification confirmée en appui par Welch + Cohen d
  (`scripts/stats.py`). Une conclusion plus forte gagnerait à n ≥ 10.
- **Instabilité du DQN vanilla** : le résultat « ≥ 200 » repose sur le
  best-checkpoint ; la politique finale n'est pas fiable selon le seed (§6).
  Double DQN (testé, §10) ne change pas le pic ; le soft-target (testé)
  **ne corrige pas l'instabilité finale** seul. Le correctif reste à
  trouver — best-checkpoint nécessaire entre-temps.
- **Mode d'échec résiduel** : la fraction de `truncated` (hover) indique que
  l'agent stationne parfois sans se poser — limite de la fonction de reward,
  pas de l'algorithme.
- **Évaluation greedy sans vent** : conforme au sujet ; la robustesse au vent
  (`enable_wind`) est un axe d'extension non couvert ici.

## 13. Travaux futurs

Par priorité décroissante d'impact attendu : (1) **stabilité finale** —
soft-target seul testé sans succès ; tester **soft + Double DQN combinés**,
voire reward/gradient clipping plus agressif, validés sur `final.pt` ;
(2) **PER** pour accélérer et stabiliser l'apprentissage ; (3) **n-step returns** ;
(4) robustesse **au vent** ; (5) passage en **continuous** (SAC) ;
(6) augmentation de n_seeds + tests statistiques formels. Chacun est isolé
architecturalement (un fichier + une clé de config) et n'impacte ni
`trainer.py` ni `evaluator.py`.

## 14. Conclusion

Un DQN vanilla, soigneusement instrumenté et reproductible, **résout
`LunarLander-v3` au-delà du critère** (267.21 ≥ 200, 5/5 seeds), à condition
de gérer son **instabilité tardive** par best-checkpoint. Les ablations
contrôlées montrent une hiérarchie claire des facteurs : la **taille du replay
buffer** a un effet significatif, la **profondeur du réseau** non. Tous les
résultats sont chiffrés, agrégés sur 5 seeds avec IC95, traçables aux runs
committés, et régénérables par une seule commande.

## Reproductibilité

`bash reproduce.sh` régénère les 15 runs (seeds 0–4), les `summary_aggregated.
json` et toutes les figures (`assets/plots/`) à ±5 % des valeurs ci-dessus.
Chaque chiffre trace vers `runs/<run>/<seed>/summary.json` (per-seed) et
`runs/<run>/summary_aggregated.json` (agrégé). Tests : `python -m unittest
discover tests/` (218 tests, vert).

## Références

[1] Mnih et al., *Human-level control through deep reinforcement learning*,
Nature 2015 (DQN). [2] Towers et al., *Gymnasium*, Farama Foundation, 2024.
[3] van Hasselt et al., *Deep RL with Double Q-learning*, AAAI 2016.
[4] Sutton & Barto, *Reinforcement Learning: An Introduction*, 2e éd., 2018.

## Annexe A — Données par seed (best-checkpoint, 100 ép. greedy)

| Run | Seed | Retour | σ | first_solve | best_step |
|---|---|---|---|---|---|
| baseline | 0 | 259.53 | 48.0 | 110 000 | 320 000 |
| baseline | 1 | 274.63 | 28.6 | 140 000 | 400 000 |
| baseline | 2 | 273.00 | 32.4 | 90 000 | 440 000 |
| baseline | 3 | 279.75 | 27.1 | 110 000 | 500 000 |
| baseline | 4 | 249.13 | 39.9 | 210 000 | 340 000 |
| buffer10k | 0 | 235.48 | 97.0 | 80 000 | 500 000 |
| buffer10k | 1 | 243.14 | 85.7 | 90 000 | 340 000 |
| buffer10k | 2 | 235.37 | 74.0 | 90 000 | 460 000 |
| buffer10k | 3 | 252.41 | 66.0 | 110 000 | 380 000 |
| buffer10k | 4 | 245.60 | 67.8 | 100 000 | 220 000 |
| depth3 | 0 | 277.30 | 19.8 | 160 000 | 210 000 |
| depth3 | 1 | 256.36 | 66.5 | 130 000 | 270 000 |
| depth3 | 2 | 279.29 | 38.7 | 80 000 | 280 000 |
| depth3 | 3 | 282.57 | 30.1 | 90 000 | 410 000 |
| depth3 | 4 | 275.98 | 53.3 | 150 000 | 460 000 |
