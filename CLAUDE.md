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
| `evaluator.py` | `BaselineEvaluator` : cache de tables + appel `Evaluator().eval()` |
| `logger.py` | `setup_logger()` JSON + `ResultsWriter` JSONL |

### Schéma `AgentResult` (SER-compatible dès Semaine 1)

```python
@dataclass
class AgentResult:
    answer: str
    generated_code: str
    raw_response: str
    confidence: Optional[float] = None   # réservé — SER Semaine 2
    abstained: Optional[bool] = None     # réservé — SER Semaine 2
```

### Schéma JSONL des résultats (un enregistrement par question)

```json
{
  "run_id": "smoke_20260605T123456Z",
  "model_id": "claude-haiku-4-5-20251001",
  "temperature": 0.0,
  "question_idx": 0,
  "question": "...",
  "dataset": "001_Forbes",
  "answer_type": "boolean",
  "ground_truth": "True",
  "answer": "True",
  "generated_code": "result = ...",
  "confidence": null,
  "abstained": null,
  "ts": "2026-06-05T12:34:56Z"
}
```

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
scripts/
  run_baseline.py       # point d'entrée : --smoke (10 q) ou full (50 q)
configs/
  baseline.toml         # modèle épinglé, temperature=0.0, tailles d'échantillon
runs/                   # résultats JSONL par run (gitignored, .gitkeep présent)
tests/
  test_pipeline.py      # tests mockés (gratuits) + live (VERIDATA_LIVE_TESTS=1)
pyproject.toml
.env.example
```

---

## Feuille de route

### ✅ Semaine 1 — Fondations + baseline (TERMINÉE)
- Scaffolding, environnement, dépendances
- Agent code-as-reasoning minimal
- Précision de base sur 50 questions DataBench
- Schema `AgentResult` SER-compatible (`confidence`/`abstained` = None)
- Tests mockés (sans coût) + live gated par `VERIDATA_LIVE_TESTS=1`
- Smoke run (10 questions) pour validation pipeline

### ⬜ Semaine 2 — Métrique SER
- Extraire la confiance depuis le LLM (auto-évaluation ou logprobs si dispo)
- Définir le seuil "haute confiance" (ex. > 0.8)
- Implémenter le calcul SER sur les fichiers JSONL de `runs/`
- Comparer SER vs précision sur différents sous-ensembles de questions
- Mettre à jour `AgentResult.confidence` et `AgentResult.abstained`

### ⬜ Semaine 3 — Perturbations
- Perturbations contrôlées : bruit dans les données, colonnes renommées, outliers
- Mesurer la dégradation SER vs précision sous perturbation
- Hardening du sandbox : passer à une exécution subprocess isolée
- Analyse de sensibilité : quels types de questions/tables sont les plus fragiles ?

### ⬜ Semaine 4 — Couche de vérification
- Mécanisme d'abstention explicite (l'agent peut dire "je ne sais pas")
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

# Tests mockés uniquement (gratuits)
pytest

# Tests live (nécessite clé API)
$env:VERIDATA_LIVE_TESTS = "1"
pytest
```
