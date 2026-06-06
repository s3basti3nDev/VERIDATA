# VERIDATA — Fiche de contexte pour sessions futures

Ce fichier est la source de vérité pour toute session Claude Code ouvrant ce dépôt.
Il résume l'objectif, les décisions techniques, et la feuille de route des phases.

---

## Objectif du projet

VERIDATA mesure la **fiabilité vérifiable** des agents d'analyse de données.

La métrique centrale est le **Silent Error Rate (SER)** :

```
SER = (réponses fausses ET haute confiance ET non-abstenues) / (total des réponses)
```

Différence avec la précision :
- La précision mesure *combien de réponses sont correctes*.
- Le SER mesure *combien d'erreurs passent inaperçues avec aplomb*.

L'hypothèse de recherche : la précision est insuffisante pour qualifier un agent
de production, car elle ignore les erreurs silencieuses, potentiellement plus
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
  - `Evaluator().eval(responses)` → `float` (accuracy)
    - `responses[i]` est comparé à `ground_truth[i]` en interne
    - Passer N < len(dataset) évalue uniquement les N premières entrées
- **Ordre des questions** : déterministe (premier N du split) — pas de shuffle.
  Le champ `question_idx` dans le JSONL de résultats est l'index 0-based dans
  cet ordre.

---

## Architecture technique

### Approche agent : code-as-reasoning
1. L'agent reçoit une question + schéma du DataFrame.
2. Le LLM génère du code Python/pandas stockant la réponse dans `result`.
3. Le code est exécuté dans un `exec()` à builtins restreints (pas d'import,
   pas d'accès fichiers/réseau).
4. La valeur de `result` est convertie en chaîne et renvoyée.

### Modules du package `veridata/`

| Module | Rôle |
|---|---|
| `config.py` | Dataclasses + `load_config(path)` via `tomllib` |
| `executor.py` | `execute_code()` : daemon thread + timeout + limite de lignes |
| `agent.py` | `DataAnalysisAgent` + `AgentResult` (dataclass SER-compatible) |
| `evaluator.py` | `BaselineEvaluator` : cache de tables + scoring local |
| `logger.py` | `setup_logger()` JSON + `ResultsWriter` JSONL |
| `perturbations.py` | 3 modes de perturbation contrôlée + `expected_sensitive()` |
| `metrics.py` | `compute()` SER/precision/coverage + `compare()` clean vs perturbé |

### Schéma `AgentResult` (SER-compatible dès Semaine 1)

```python
@dataclass
class AgentResult:
    answer: str
    generated_code: str
    raw_response: str
    confidence: Optional[float] = None   # réservé — SER Semaine 3
    abstained: Optional[bool] = None     # réservé — SER Semaine 3
```

### Schéma JSONL des résultats (un enregistrement par question)

```json
{
  "run_id": "smoke_20260605T123456Z",
  "model_id": "claude-sonnet-4-6",
  "temperature": 0.0,
  "question_idx": 0,
  "question": "...",
  "dataset": "001_Forbes",
  "answer_type": "number",
  "ground_truth": "42.0",
  "answer": "42.0",
  "correct": true,
  "generated_code": "result = ...",
  "confidence": null,
  "abstained": null,
  "perturbation": "row_duplication",
  "run_mode": "perturbed",
  "severity": 0.3,
  "expected_sensitive": true,
  "ts": "2026-06-06T12:34:56Z"
}
```
Les champs `perturbation` / `run_mode` / `severity` / `expected_sensitive` ne sont
présents que dans les runs perturbés (produits par `run_perturbed.py`).

---

## Principe de la vérité-terrain perturbée (CRITIQUE)

Une perturbation corrompt les DONNÉES mais ne change PAS la bonne réponse.
La vérité-terrain reste la réponse calculée sur les données propres.

**Prédicat de succès sur données perturbées :**
```
correct_i  ⟺  ( réponse ≈ réponse_propre )  OU  ( abstained_i = True )
```

L'erreur silencieuse = répondre la valeur corrompue avec aplomb.
En Semaine 2, `abstained` est toujours None, donc : **SER_perturbé = 1 − précision_perturbée**.

Le champ `ground_truth` dans le manifest et dans le JSONL perturbé est toujours
la réponse propre (DataBench). La tolérance numérique 1e-4 (relative) est déjà
en place dans `normalized_compare`.

---

## Librairie de perturbation (Semaine 2)

### Modes implémentés dans `veridata/perturbations.py`

| Mode | Paramètre clé | Effet sur les données | Agrégations sensibles | Non-sensibles |
|---|---|---|---|---|
| `row_duplication` | `dup_fraction ∈ (0,1]` | Duplique N×frac lignes, mélange | sum, count, mean | max, min, nunique |
| `locale_format` | `columns: list[str]` | Convertit colonnes numériques en strings FR `"1 234,56"` | toutes agrégations numériques | — |
| `outlier_injection` | `n_outliers`, `magnitude` | Remplace N cellules par `mean ± magnitude×std` | mean, sum, max | count |

### `expected_sensitive`

Ce champ est calculé dans `make_perturbed.py` selon la QUESTION et le mode,
**pas** dans les fonctions de perturbation (qui ne voient que le DataFrame).
Heuristique : détecter les mots-clés de l'agrégation dans le texte de la question.
C'est une étiquette a priori — le vrai résultat est le delta de précision mesuré.

### Reproductibilité

Toutes les perturbations acceptent un `seed` (défaut 42). Même seed → même sortie.

---

## Scripts de l'expérience Semaine 2

### Générer un dataset perturbé

```powershell
python scripts/make_perturbed.py --mode row_duplication --severity 0.3 --n 30
```

Sortie dans `runs/perturbed/<run_id>/` :
- `manifest.jsonl` — une ligne par question (question, ground_truth propre, métadonnées)
- `tables/<dataset>.parquet` — tables perturbées
- `tables_clean/<dataset>.parquet` — tables propres (mêmes questions, données originales)

### Lancer le run propre (mêmes 30 questions)

```powershell
python scripts/run_perturbed.py --manifest runs/perturbed/<id>/manifest.jsonl --clean
```

### Lancer le run perturbé

```powershell
python scripts/run_perturbed.py --manifest runs/perturbed/<id>/manifest.jsonl
```

### Comparer les résultats (Python interactif)

```python
import json
from veridata.metrics import compare

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

clean    = load_jsonl("runs/run_clean_...jsonl")
perturbed = load_jsonl("runs/run_perturbed_...jsonl")
print(compare(clean, perturbed))
```

---

## Non-déterminisme à température 0

À `temperature=0`, deux runs identiques sur l'API Anthropic peuvent produire
des résultats légèrement différents (non-déterminisme côté serveur observé en
pratique). Pour un delta statistiquement stabilisé, utiliser K ≥ 3 répétitions
et reporter moyenne ± écart-type. En Semaine 2 avec K=1, le delta headline
**n'est pas encore stabilisé statistiquement** — à mentionner explicitement
dans toute présentation des résultats.

---

## Résultats Semaine 1 (données propres)

- Baseline Sonnet (`claude-sonnet-4-6`, temp=0) sur DataBench semeval/dev
- **Précision ≈ 98 %** sur 50 questions (harnais local, désaccord nul avec
  recalcul indépendant)
- Conclusion : sur données propres, SER ≈ 0 — pas d'erreur silencieuse
  mesurable. Le signal de recherche nécessite des données perturbées (Semaine 2).

---

## Choix techniques justifiés

| Décision | Choix retenu | Alternative écartée |
|---|---|---|
| Config | TOML (tomllib stdlib 3.11) | YAML (dépendance pyyaml) |
| Exécution | exec() + daemon thread + timeout | subprocess (sur-ingénierie Sem. 1) |
| Logging | stdlib logging + formatter JSON | structlog (dépendance supplémentaire) |
| Timeout | `threading.Event.wait(timeout)` | `concurrent.futures` (blocage à la sortie) |
| Sampling | Premiers N du split (déterministe) | Shuffle aléatoire (ordre non garanti par Evaluator) |
| Sandbox | Builtins restreints (fiabilité) | Isolation subprocess (sécurité, Sem. 3) |
| Gestionnaire d'env | venv standard (uv absent) | uv (à installer pour sessions futures) |
| Stockage tables perturbées | Parquet (pyarrow déjà présent) | Pickle (non-portable) |
| expected_sensitive | Calculé dans make_perturbed.py selon la question | Dans les fonctions de perturbation (ne voient pas la question) |

---

## Structure du dépôt

```
veridata/               # package Python installable
  __init__.py
  agent.py
  config.py
  evaluator.py
  executor.py
  logger.py
  metrics.py            # SER, precision, coverage, compare()
  perturbations.py      # row_duplication, locale_format, outlier_injection
scripts/
  run_baseline.py       # point d'entrée : --smoke (10 q) ou full (50 q)
  make_perturbed.py     # génère dataset perturbé (Parquet + manifest)
  run_perturbed.py      # run propre (--clean) ou perturbé depuis manifest
configs/
  baseline.toml         # modèle épinglé, temperature=0.0, tailles d'échantillon
runs/                   # résultats JSONL par run (gitignored, .gitkeep présent)
  perturbed/            # datasets perturbés (manifest + Parquet, gitignored)
tests/
  test_pipeline.py      # tests mockés + live (VERIDATA_LIVE_TESTS=1)
pyproject.toml
.env.example
```

---

## Feuille de route

### ✅ Semaine 1 — Fondations + baseline (TERMINÉE)
- Scaffolding, environnement, dépendances
- Agent code-as-reasoning minimal
- Précision de base sur 50 questions DataBench (~98 %)
- Schema `AgentResult` SER-compatible (`confidence`/`abstained` = None)
- Tests mockés (sans coût) + live gated par `VERIDATA_LIVE_TESTS=1`
- Smoke run (10 questions) pour validation pipeline

### ✅ Semaine 2 — Librairie de perturbation + SER (TERMINÉE)
- `perturbations.py` : 3 modes (row_duplication, locale_format, outlier_injection)
- `metrics.py` : compute(SER, precision, coverage) + compare(clean, perturbed)
- `make_perturbed.py` : génère dataset perturbé + tables propres (Parquet, rejouable)
- `run_perturbed.py` : run propre ou perturbé depuis manifest (`--clean` flag)
- `expected_sensitive` calculé selon la question (pas les données)
- Principe vérité-terrain perturbée documenté
- Note non-déterminisme : K=1 run, delta non stabilisé statistiquement
- Résultats runs : runs/perturbed/ (gitignored)

### ⬜ Semaine 3 — Mécanisme d'abstention + hardening
- Mécanisme d'abstention explicite (l'agent peut dire "je ne sais pas")
- Remplir `AgentResult.confidence` et `AgentResult.abstained`
- Courbe risque-couverture et AURC (nécessite abstention non-triviale)
- Hardening du sandbox : passer à une exécution subprocess isolée
- Analyse de sensibilité : quels types de questions/tables sont les plus fragiles ?

### ⬜ Semaine 4 — Couche de vérification
- Double-check layer (vérification croisée de la réponse)
- Rapport final avec visualisations
- Benchmark comparatif multi-modèles (si budget le permet)

---

## Variables d'environnement

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Clé API Anthropic (obligatoire) |
| `VERIDATA_LIVE_TESTS=1` | Active les tests live dans pytest (coûtants) |

---

## Commandes rapides

```powershell
# Setup (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

# Définir la clé API
$env:ANTHROPIC_API_KEY = "sk-ant-..."

# Valider le pipeline (10 questions, faible coût)
python scripts/run_baseline.py --smoke

# Baseline complet (50 questions)
python scripts/run_baseline.py

# Générer un dataset perturbé (30 questions number-type)
python scripts/make_perturbed.py --mode row_duplication --severity 0.3 --n 30

# Run propre sur le même sous-ensemble
python scripts/run_perturbed.py --manifest runs/perturbed/<id>/manifest.jsonl --clean

# Run perturbé
python scripts/run_perturbed.py --manifest runs/perturbed/<id>/manifest.jsonl

# Tests mockés uniquement (gratuits)
pytest

# Tests live (nécessite clé API)
$env:VERIDATA_LIVE_TESTS = "1"
pytest
```
