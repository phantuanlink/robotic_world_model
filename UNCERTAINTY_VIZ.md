# Uncertainty Visualization — Guide de lecture

Script : `run_uncertainty_viz.py`  
Checkpoint : `checkpoints_paws/model_2000_pretrain.pt`  
Données : `utils_paws/walk_spot2.npz`

---

## 1. Le modèle : SystemDynamicsEnsemble

Le world model est un **GRU ensembliste** entraîné à prédire le prochain état du robot Go2 à partir de son historique d'états et d'actions.

```
Input à chaque step :
  état normalisé   s_t  ∈ ℝ^45
  action normalisée a_t  ∈ ℝ^12
  → concaténés en x_t ∈ ℝ^57

Backbone partagé (identique pour toutes les têtes) :
  GRU  :  57 → hidden 256,  2 couches
  (les poids du GRU sont communs à toutes les têtes)

5 têtes indépendantes (h0 … h4), chacune :
  MLP mean    :  256 → 128 → 45   →  μ_e(t)   moyenne prédite du prochain état
  MLP logstd  :  256 → 128 → 45   →  σ_e(t)   écart-type aléatoire par dimension
  (les poids des MLP sont différents par tête)
```

### Pourquoi un ensemble de têtes ?

Chaque tête a été initialisée différemment et entraînée simultanément.
Si elles prédisent des valeurs proches → le modèle est **confiant**.
Si elles divergent → le modèle est **incertain** (état peu vu à l'entraînement).

---

## 2. Les deux types d'incertitude

### 2.1 Incertitude épistémique (bleue)

> "Ce que le modèle ne sait pas."

Calculée comme **l'écart-type des moyennes prédites à travers les 5 têtes**, sommé sur les 45 dimensions d'état :

```
σ_epi(t) = Σ_d  std_{e=0..4}( μ_e(t, d) )
```

| Valeur | Interprétation |
|--------|----------------|
| Faible | Le robot est dans un état bien couvert par les données d'entraînement |
| Élevée | État inhabituel, hors distribution, ou transition physique complexe |

**Peut être réduite** en ajoutant plus de données d'entraînement diversifiées.

---

### 2.2 Incertitude aléatoire (rouge)

> "Le bruit irréductible du système physique."

Calculée comme la **moyenne des écarts-types prédits par chaque tête**, sommé sur les 45 dimensions :

```
σ_ale(t) = Σ_d  mean_{e=0..4}( σ_e(t, d) )
```

| Valeur | Interprétation |
|--------|----------------|
| Faible | La dynamique est localement prévisible |
| Élevée | Contact, glissement, perturbation externe — le futur est intrinsèquement variable |

**Ne peut pas être réduite** avec plus de données : c'est le bruit du monde réel.

---

## 3. L'espace d'état (45 dimensions)

| Indices | Variable | Unité | Description |
|---------|----------|-------|-------------|
| 0–2     | `vx vy vz`   | m/s   | Vitesse linéaire de la base |
| 3–5     | `wx wy wz`   | rad/s | Vitesse angulaire de la base |
| 6–8     | `gx gy gz`   | —     | Gravité projetée dans le repère corps (≈ orientation) |
| 9–20    | `q0 … q11`   | rad   | Positions articulaires (12 joints) |
| 21–32   | `dq0 … dq11` | rad/s | Vitesses articulaires |
| 33–44   | `τ0 … τ11`   | Nm    | Couples articulaires |

Ordre des joints : FR_Hip, FR_Thigh, FR_Calf, FL_Hip, FL_Thigh, FL_Calf, RR_Hip, RR_Thigh, RR_Calf, RL_Hip, RL_Thigh, RL_Calf.

Les données sont **normalisées** avant passage dans le modèle :
```
s_norm = (s_raw - mean) / std
```
avec `mean[8] = -1.0` (gravité z au repos), `std` typique : 0.5 m/s pour les vitesses, 0.2 rad pour les positions.

---

## 4. Lecture de la figure (3 lignes × 6 colonnes)

```
┌─────────────────┬─────────────────┬─────────────────┐
│ Histogramme     │ Histogramme     │ Série temporelle │  ← Ligne 1
│ σ_épistémique   │ σ_aléatoire     │ des deux unc.    │
├────┬────┬────┬──┴─┬────┬──────────┴──────────────────┤
│ μ  │ μ  │ μ  │ μ  │ μ  │ μ                           │  ← Ligne 2
│ vx │ vy │ vz │ wx │ wy │ wz  (6 premières dims)       │
├────┬────┬────┬────┬────┬─────────────────────────────┤
│ σ  │ σ  │ σ  │ σ  │ σ  │ σ                           │  ← Ligne 3
│ vx │ vy │ vz │ wx │ wy │ wz  (mêmes dims)             │
└────┴────┴────┴────┴────┴─────────────────────────────┘
```

---

### Ligne 1 — Vue globale

#### Histogramme σ_épistémique (gauche)
- Axe X : valeur de l'incertitude épistémique (somme sur 45 dims)
- Hauteur de barre : nombre de timesteps à cette valeur
- **Queue à droite** = quelques moments où le modèle est très incertain
- Distribution concentrée = comportement homogène sur tout le rollout

#### Histogramme σ_aléatoire (centre)
- Même lecture que ci-dessus pour l'incertitude aléatoire
- Généralement plus concentré (bruit physique relativement constant)

#### Séries temporelles (droite)
- Axe X : numéro de step (après les 32 steps de warm-up GRU)
- Axe Y : valeur d'incertitude
- **Pics bleus** = instants où le robot est dans un état hors distribution
- **Pics rouges** = instants physiquement bruités (atterrissages, glissements)
- Les deux pics coïncident souvent mais pas toujours

---

### Ligne 2 — Moyennes prédites μ par tête

Chaque sous-plot correspond à une dimension de l'état (vx, vy, vz, wx, wy, wz).

- **5 courbes colorées** (h0–h4) = la prédiction du prochain état par chaque tête
- **Courbes confondues** = modèle confiant sur cette dimension
- **Courbes qui s'écartent** = incertitude épistémique élevée sur cette dimension
- L'axe Y est en **espace normalisé** (sans unité) : 0 ≈ valeur moyenne, ±1 ≈ ±1 écart-type

---

### Ligne 3 — Écarts-types prédits σ par tête

Même layout que la ligne 2, mais chaque courbe montre **l'incertitude aléatoire de la tête** sur cette dimension.

- **σ élevé** sur une dim = la tête dit "même avec la même entrée, la sortie est variable"
- Les 5 têtes ont typiquement des σ similaires (même backbone GRU partagé)
- Un écart entre têtes sur σ indique que les têtes ont appris des "niveaux de confiance" différents pour cette dimension

---

## 5. Warmup GRU

Le GRU a besoin de **32 steps d'historique** pour initialiser son état caché avant de faire des prédictions fiables. Le script consomme donc les 32 premiers frames pour le warm-up et commence la visualisation à `t = 32`.

Sur 3 077 frames totales → **3 045 frames visualisées**.

---

## 6. Ce que l'incertitude révèle sur la marche

| Observation dans les plots | Cause probable |
|---------------------------|----------------|
| Pic d'incertitude épistémique | Transition de foulée, changement de vitesse, sol irrégulier |
| σ_aléatoire élevé sur τ (couples) | Contacts pied-sol avec glissement ou rebond |
| σ_aléatoire élevé sur dq (vitesses) | Oscillations articulaires rapides après contact |
| Têtes très divergentes sur vx/vy | Accélération ou décélération peu vue à l'entraînement |
| Incertitude globalement faible | Phase de marche stable et régulière |

---

## 7. Commande

```powershell
cd "C:\Users\tjga9\Documents\Tomas\EPFL\AI Team\RWM"
& "C:\Users\tjga9\anaconda3\python.exe" run_uncertainty_viz.py `
    --checkpoint checkpoints_paws/model_2000_pretrain.pt `
    --data       utils_paws/walk_spot2.npz `
    --device     cpu `
    --pause      0.05 `
    --update_every 10
```

### Paramètres utiles

| Paramètre | Défaut | Effet |
|-----------|--------|-------|
| `--update_every N` | 10 | Redessine tous les N steps (plus grand = plus rapide) |
| `--pause P` | 0.05 | Secondes d'attente entre chaque frame |
| `--history H` | 32 | Taille de la fenêtre de warm-up GRU |
| `--device cuda` | cpu | Utilise le GPU si disponible |
