# Rapport (brouillon) — Start Trek / LunarLander DQN

> Brouillon de travail. Tous les chiffres tracent vers des runs commitées
> (`runs/<run>/<seed>/summary.json`, agrégés via `scripts/aggregate_runs.py`).
> Le rapport PDF final est T4.2.

## Baseline DQN (référence, 5 seeds)

`configs/baseline/dqn.yaml`, seeds 0..4, 500k steps, best-checkpoint.

- **mean return (100 ép. greedy, agrégé 5 seeds) : 267.21, IC95 [256.19, 278.22]**
- solved : **5/5** (seuil ≥ 200)
- `first_solve_step` moyen : 132 000 steps (IC95 [90.7k, 173.3k])
- Instabilité tardive systématique du DQN vanilla : la politique finale
  s'effondre sur chaque seed (ex. seed 0 : éval step 500k = -604) ; le
  best-checkpoint capture le pic (~step 250k) — d'où l'écart entre
  `mean_return_last_100` (best) et la politique finale.

## Ablation A — Taille du replay buffer (10k vs 100k)

**Choix de l'ablation** : taille du replay buffer, **seul** paramètre modifié
(`buffer_size : 100000 → 10000` ; tout le reste identique, vérifié par diff de
config). Motivation : hypothèse standard qu'un buffer plus petit ⇒ échantillons
plus corrélés et oubli plus rapide ⇒ apprentissage plus instable.
`configs/ablations/dqn_buffer10k.yaml`, seeds 0..4.

| Métrique (5 seeds, agrégé) | Baseline (100k) | Ablation (10k) |
|---|---|---|
| mean return, IC95 | **267.21** [256.19, 278.22] | **242.40** [236.08, 248.72] |
| solved | 5/5 | 5/5 |
| `first_solve_step` moyen | 132 000 | ~94 000 |

**Analyse.** Réduire le buffer d'un facteur 10 dégrade la performance
asymptotique de façon **statistiquement significative** : les intervalles de
confiance à 95 % sont **disjoints** (borne basse baseline 256.19 > borne haute
ablation 248.72), soit ≈ **−25 points (−9 %)** de return moyen. Le petit buffer
reste néanmoins suffisant pour *résoudre* LunarLander (5/5 seeds ≥ 200).
Contre-intuitivement, l'ablation atteint le seuil de 200 **plus tôt**
(`first_solve_step` ~94k vs 132k) : avec un buffer plus petit, les transitions
échantillonnées reflètent une politique plus récente (moins de retard de
distribution), ce qui accélère la montée initiale — mais au prix d'un plafond
plus bas et d'une variance inter-seed plus élevée (échantillons plus corrélés,
oubli catastrophique plus marqué). Le compromis est clair : **buffer large =
apprentissage plus lent à démarrer mais plateau plus haut et plus stable**.

Figure : `assets/plots/paired_buffer10k_vs_baseline.png` (courbes de return vs
steps, moyenne ± IC95 sur 5 seeds, baseline vs ablation).

## Ablation B

_À compléter — T3.4._
