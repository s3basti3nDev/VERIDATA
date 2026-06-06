# VERIDATA — Fiche de contexte pour sessions futures

Ce fichier est la source de vérité pour toute session Claude Code ouvrant ce dépôt.
Il résume l'objectif, les décisions techniques, et la feuille de route des phases.

---

## Objectif du projet

VERIDATA mesure la **fiabilité vérifiable** des agents d'analyse de données.

La métrique centrale est le **Silent Error Rate (SER)** :

```
SER = (réponses fausses ET non-abstenues) / (total des réponses)
```

Différence avec la précision :
- La précision mesure *combien de réponses sont correctes*.
- Le SER mesure *combien d'erreurs passent inaperçues sans alerte*.

L'hypothèse de recherche : la précision est insuffisante pour qualifier un agent
de production, car elle ignore les erreurs silencieuses — potentiellement plus
dangereuses que les erreurs avouées.

---

## Dataset de référence

- **Source** : `cardiffnlp/databench` (HuggingFace)
- **Taille** : ~1822 questions, 80 tables réelles
- **Outillage** : package PyPI `databench-eval` (≥ 4.0)
- **API** :
  - `load_qa(name="semeval", split="dev")` → `list[dict]`
    - Clés : `question`, `answer`, `type`, `dataset`, `columns_used`, `sample_answer`
  - `load_table(dataset_id)` → `pd.DataFrame`
- **Ordre des questions** : déterministe (premier N du split) — pas de shuffle.

---

## Architecture technique

### Approche agent : code-as-reasoning
1. L'agent reçoit une question + schéma du DataFrame.
2. Le LLM génère du code Python/pandas stockant la réponse dans `result`.
3. Le code est exécuté dans un `exec()` à builtins restreints.
4. La valeur de `result` est convertie en chaîne et renvoyée.
5. **(Semaine 3)** : la couche d'invariants vérifie le code et les données AVANT
   de valider la réponse. Si un invariant fire → abstained=True.

### Modules du package `veridata/`

| Module | Rôle |
|---|---|
| `config.py` | Dataclasses (dont `InvariantsConfig`) + `load_config(path)` |
| `executor.py` | `execute_code()` : daemon thread + timeout + limite de lignes |
| `agent.py` | `DataAnalysisAgent` + `AgentResult` |
| `evaluator.py` | `BaselineEvaluator` : cache de tables + scoring local |
| `logger.py` | `setup_logger()` JSON + `ResultsWriter` JSONL |
| `perturbations.py` | 3 modes + `expected_sensitive()` |
| `metrics.py` | `compute()` SER/precision/coverage/sur_abstention + `compare()` |
| `invariants.py` | 4 invariants agnostiques à la réponse + `InvariantResult` |
| `verifier.py` | `run_invariants()` → `VerificationResult` (abstained, confidence, trace) |

### Schéma `AgentResult`

```python
@dataclass
class AgentResult:
    answer: str
    generated_code: str
    raw_response: str
    confidence: Optional[float] = None   # rempli par verifier.py (Sem. 3)
    abstained: Optional[bool] = None     # rempli par verifier.py (Sem. 3)
```

### Schéma JSONL étendu (run vérifié)

```json
{
  "run_id": "verified_row_duplication_0.3_20260606T...",
  "model_id": "claude-sonnet-4-6",
  "question_idx": 0,
  "question": "...",
  "ground_truth": "42.0",
  "answer": "54.6",
  "correct": false,
  "abstained": true,
  "confidence": 0.2,
  "invariants": [
    {"name": "duplicate_rows", "fired": true,  "severity": 0.6, "detail": "dup_fraction=0.30..."},
    {"name": "numeric_outliers","fired": false, "severity": 0.0, "detail": "no values beyond k=5×IQR"},
    {"name": "dtype_mismatch",  "fired": false, "severity": 0.0, "detail": "all columns numeric"},
    {"name": "unexplained_constant","fired": false, "severity": 0.0, "detail": "no candidate constants"}
  ],
  "perturbation": "row_duplication",
  "run_mode": "row_duplication",
  "severity": 0.3,
  "expected_sensitive": true,
  "ts": "2026-06-06T..."
}
```

---

## Principe de la vérité-terrain perturbée (CRITIQUE)

Une perturbation corrompt les DONNÉES mais ne change PAS la bonne réponse.

**Prédicat de succès sur données perturbées :**
```
correct_i  ⟺  ( réponse ≈ réponse_propre )  OU  ( abstained_i = True )
```

- `correct` dans le JSONL = vérité factuelle (`normalized_compare`)
- Le SER = `mean(not correct AND not abstained)` intègre l'abstention automatiquement

**Métriques Semaine 3 :**
```
SER_sans_invariants = 1 − précision              (run_perturbed.py, abstained=None)
SER_avec_invariants = mean(wrong AND not abstained)   (run_verified.py)
coverage            = mean(not abstained)
sur_abstention      = mean(correct AND abstained)     # faux positifs d'abstention
false_positive_rate = mean(abstained) sur run clean   # ≈ 1 − coverage sur clean
```

---

## Invariants (Semaine 3)

### Principe directeur

Un invariant ne calcule JAMAIS la réponse. Sa sortie est binaire (fired/non-fired)
+ un détail traçable.

**Contrainte anti-confond** : les invariants sont des contrôles data-quality
LÉGITIMES, justifiables indépendamment de nos perturbations.

### Politique d'abstention (révisée après mesure empirique)

Seuls les invariants avec pouvoir discriminant prouvé déclenchent l'abstention.
`numeric_outliers` est **trace-only** : sur les tables DataBench réelles, il fire
à taux quasi-identique sur données propres et perturbées (delta ≈ 0), ce qui
dégrade la précision de l'abstention sans améliorer le recall.

**Raison fondamentale** : un outlier injecté est statistiquement indistinguable
d'un outlier réel (followers Twitter, revenus d'entreprise, scores sportifs).

| Invariant | Cible perturbation | Déclenche abstention |
|---|---|---|
| `duplicate_rows` | row_duplication | ✅ OUI |
| `numeric_outliers` | outlier_injection | ❌ TRACE ONLY |
| `dtype_mismatch` | locale_format | ✅ OUI |
| `unexplained_constant` | hallucination LLM | ✅ OUI |

### Fallback conservateur pour indirection

Quand l'AST détecte un subscript dynamique (variable en clé plutôt que littéral),
`numeric_outliers` et `dtype_mismatch` élargissent leur scope à TOUTES les colonnes
et inscrivent `scope=all_columns, reason=indirection` dans le détail.

### Les quatre invariants — détail

| Invariant | Condition de déclenchement |
|---|---|
| `duplicate_rows` | dup_fraction > threshold (0.05) ET agg sensible dans code |
| `numeric_outliers` | valeurs au-delà k×IQR (k=5) dans colonnes référencées |
| `dtype_mismatch` | colonne non-numérique utilisée dans une agg numérique |
| `unexplained_constant` | littéral float dans BinOp (×÷+−), absent données+question |

**`unexplained_constant` — périmètre précis** : seuls les opérandes de BinOp
(Mult/Div/Add/Sub) sont ciblés. Les constantes dans les Compare (seuils légitimes,
ex. `df['ratio'] > 0.5`) et les Subscript (indices) sont EXCLUES via l'AST.

### Seuils configurables (`baseline.toml [invariants]`)

```toml
duplicate_row_threshold = 0.05   # sensible à partir de 5 % de doublons exacts
outlier_iqr_factor      = 5.0   # strict pour limiter faux positifs sur données réelles
trivial_constants       = [0, 1, 2, -1, 100, 0.5]
```

### Limite documentée (à ne pas masquer)

Aucun invariant ne couvre la classe **"sémantique plausible + grounded"** :
une valeur extraite d'une table de référence obsolète, dont la provenance est
propre et la valeur crédible. VERIDATA réduit le SER sur les classes couvertes
(données corrompues, mauvais types, doublons, constantes halluccinées) ; il ne
prétend pas l'annuler sur toutes les classes d'erreur possibles.

---

## Librairie de perturbation (Semaine 2)

| Mode | Paramètre clé | Agrégations sensibles | Non-sensibles |
|---|---|---|---|
| `row_duplication` | `dup_fraction ∈ (0,1]` | sum, count, mean | max, min, nunique |
| `locale_format` | `columns: list[str]` | toutes agrégations numériques | — |
| `outlier_injection` | `n_outliers`, `magnitude` | mean, sum, max | count |

**Résultats Semaine 2** (K=1, non stabilisé, 30 questions number-type, severity=0.3) :

| Mode | Précision clean→perturbé | Erreurs |
|---|---|---|
| row_duplication | 96.7 % → 21.1 % (sensibles) | SILENCIEUSES |
| outlier_injection | 100 % → 77.8 % (sensibles) | SILENCIEUSES |
| locale_format | 96.7 % → 73.3 % | Majoritairement bruyantes + 1 silencieux (×100) |

---

## Scripts de l'expérience

### Générer un dataset perturbé

```powershell
python scripts/make_perturbed.py --mode row_duplication --severity 0.3 --n 30
```

Sortie dans `runs/perturbed/<run_id>/` :
- `manifest.jsonl` — questions + ground truth propre + métadonnées
- `tables/<dataset>.parquet` — tables perturbées
- `tables_clean/<dataset>.parquet` — tables propres (mêmes questions)

### Runs Semaine 2 (sans invariants)

```powershell
python scripts/run_perturbed.py --manifest <manifest> --clean   # run propre
python scripts/run_perturbed.py --manifest <manifest>           # run perturbé
```

### Runs Semaine 3 (avec invariants)

```powershell
python scripts/run_verified.py --manifest <manifest> --clean    # faux positifs
python scripts/run_verified.py --manifest <manifest>            # SER avec invariants
```

### Afficher la trace d'une question

```powershell
python scripts/show_trace.py --run runs/verified_....jsonl --question_idx 5
```

### Tableau de discrimination par invariant (résultat headline)

```powershell
# Perturbed + clean (recommandé)
python scripts/analyze_invariants.py `
    --perturbed runs/verified_row_duplication_....jsonl `
    --clean     runs/verified_clean_....jsonl

# Perturbed seul (sans fire_rate_clean)
python scripts/analyze_invariants.py --perturbed runs/verified_....jsonl
```

Colonnes : `fire_rate_perturbed`, `fire_rate_clean`, `delta` (pouvoir discriminant),
`precision` P(wrong|fired), `recall` P(fired|wrong), `abstains` (Y / N trace).

### Comparer clean vs perturbé (Python interactif)

```python
import json
from veridata.metrics import compare

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

print(compare(load_jsonl("runs/verified_clean_...jsonl"),
              load_jsonl("runs/verified_row_duplication_...jsonl")))
```

---

## Non-déterminisme à température 0

À `temperature=0`, deux runs identiques peuvent différer (non-déterminisme côté
serveur Anthropic observé en pratique). Pour un delta headline stabilisé : K ≥ 3
répétitions, reporter moyenne ± écart-type. En K=1, le delta **n'est pas stabilisé
statistiquement** — à mentionner explicitement dans toute présentation.

---

## Résultats par semaine

### Semaine 1 — Baseline clean
- Sonnet `claude-sonnet-4-6`, temp=0, 50 questions DataBench
- **Précision ≈ 98 %** — SER ≈ 0 sur données propres
- Conclusion : le signal de recherche nécessite des données perturbées

### Semaine 2 — Perturbations (K=1)
- Voir tableau ci-dessus

### Semaine 3 — Invariants (K=3, n=30 questions number, severity=0.3)

| Mode | SER_sans | SER_avec | Coverage | FPR clean |
|---|---|---|---|---|
| row_duplication | 50.0 % | **6.7 % [2.8, 14.1]** | 35.6 % | 16.7 % |
| locale_format | 24.4 % | **12.2 % [6.8, 20.7]** | 56.7 % | 16.7 % |
| outlier_injection | 18.3 % | **18.3 %** (inchangé) | 83.3 % | 16.7 % |

Tableau de discrimination par invariant :

| Invariant | Mode cible | Δ fire_rate [IC Wilson 95 %] | Abstient |
|---|---|---|---|
| `duplicate_rows` | row_duplication | **+0.807 [0.636, 0.890]** | ✓ |
| `dtype_mismatch` | locale_format | +0.233 [0.047, 0.401] (marginal) | ✓ |
| `numeric_outliers` | outlier_injection | -0.048 [-0.248, 0.166] (IC∋0) | ✗ trace |
| `unexplained_constant` | hallucination | +0.000 (jamais déclenché) | ✓ |

---

## Choix techniques justifiés

| Décision | Choix retenu | Alternative écartée |
|---|---|---|
| Config | TOML (tomllib stdlib 3.11) | YAML (dépendance pyyaml) |
| Exécution | exec() + daemon thread + timeout | subprocess (sur-ingénierie Sem. 1) |
| Analyse code | `ast` (stdlib) | regex fragile |
| Fallback indirection | Élargir périmètre + tracer | Ignorer silencieusement |
| Seuil IQR | k=5 (strict) | k=3 (trop de faux positifs sur données réelles) |
| Stockage tables | Parquet (pyarrow présent) | Pickle (non-portable) |
| expected_sensitive | Calculé selon la question | Dans les fonctions de perturbation |
| σ vs IC Wilson | Deux incertitudes distinctes : σ_K = bruit modèle, Wilson = robustesse éch. | Un seul IC ambigu |
| Integrity check | git log invariants.py vs ts du run → refuse si modifié après | Recomputation silencieuse |

---

## Structure du dépôt

```
veridata/
  invariants.py       # 4 invariants + InvariantResult
  verifier.py         # run_invariants() + VerificationResult
  metrics.py          # compute/compare, inclut sur_abstention
  perturbations.py    # 3 modes + expected_sensitive()
  agent.py / evaluator.py / executor.py / logger.py / config.py
scripts/
  run_baseline.py     # baseline 50 questions clean
  make_perturbed.py   # génère dataset perturbé (Parquet + manifest)
  run_perturbed.py    # run sans invariants (--clean ou perturbé)
  run_verified.py     # run avec invariants (--clean pour FPR)
  show_trace.py           # affiche trace auditable pour une question
  analyze_invariants.py   # tableau de discrimination par invariant
  repeat_runs.py          # agrégation K runs + integrity check + σ vs Wilson CI
  make_figures.py         # Fig 1/2/3 PNG → report/figures/
configs/
  baseline.toml       # inclut section [invariants]
runs/                 # JSONL par run (gitignored)
  perturbed/          # manifests + Parquet (gitignored)
  agg_row_duplication_k3.json   # résultats agrégés K=3 (gitignored)
report/
  RAPPORT.md          # rapport final (chiffres réels)
  figures/            # Fig 1/2/3 PNG
tests/
  test_pipeline.py    # 89 tests mockés + 1 skipped (live)
```

---

## Feuille de route

### ✅ Semaine 1 — Fondations + baseline
### ✅ Semaine 2 — Librairie de perturbation + SER sans invariants
### ✅ Semaine 3 — Invariants agnostiques + couche de vérification
- `invariants.py` : 4 invariants + fallback indirection conservateur
- `verifier.py` : abstention + confidence + trace auditable complète
- `run_verified.py` : pipeline complet avec vérification
- `show_trace.py` : auditabilité par question
- `analyze_invariants.py` : tableau de discrimination par invariant (headline)
- Seuil k=5 pour `numeric_outliers` (justifié par données réelles)
- Périmètre `unexplained_constant` : BinOp uniquement, exclusion Compare/Subscript
- **Politique d'abstention révisée** : `numeric_outliers` trace-only (delta≈0 sur DataBench)
- Limite documentée : classe "sémantique plausible + grounded" non couverte

### ✅ Semaine 4 — Rapport final
- `repeat_runs.py` : agrégation K runs, integrity check (git invariants.py), distinctness check,
  σ_K (bruit modèle) séparé de l'IC Wilson (robustesse statistique)
- `make_figures.py` : Fig 1 (discrimination), Fig 2 (SER), Fig 3 (K=3 ± σ)
- `report/RAPPORT.md` : rapport complet, chiffres réels, figures référencées
- **Chiffre phare stabilisé K=3** : `duplicate_rows` Δ = **+0.807 ± 0.030** [IC Wilson : 0.636, 0.890]
- σ ≈ 0 confirme quasi-déterminisme du modèle à T=0 (27/30 codes identiques)
- K=2 outlier_injection : pas de σ (non significatif)
- 89 tests mockés + 1 skipped (live)

**Loi dégagée** : un invariant discrimine ⟺ sa signature de déclenchement est
structurellement distincte de la variation naturelle des données réelles.

---

## Variables d'environnement

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Clé API Anthropic (obligatoire) |
| `VERIDATA_LIVE_TESTS=1` | Active les tests live dans pytest |

---

## Commandes rapides

```powershell
# Setup
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -e ".[dev]"
$env:ANTHROPIC_API_KEY = "sk-ant-..."

# Tests mockés (gratuits)
pytest

# Générer un dataset perturbé
python scripts/make_perturbed.py --mode row_duplication --severity 0.3 --n 30

# Run avec invariants — perturbé puis clean (faux positifs)
python scripts/run_verified.py --manifest runs/perturbed/<id>/manifest.jsonl
python scripts/run_verified.py --manifest runs/perturbed/<id>/manifest.jsonl --clean

# Afficher la trace d'une question
python scripts/show_trace.py --run runs/verified_....jsonl --question_idx 5
```
