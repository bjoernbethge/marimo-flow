# marimo-flow Codebase Analysis für AI-Human Notebooks

## 📊 Überblick

marimo-flow hat einen **gut strukturierten Kern** für Physics-Informed Machine Learning + eine Sammlung praktischer **Examples** für reale Workflows.

```
src/marimo_flow/
├── core/              ← Manager-based API für PINN & ML
│   ├── ProblemManager    (Problem-Definitionen)
│   ├── ModelFactory      (Auto-Model-Selection)
│   ├── SolverManager     (Training & Solvers)
│   └── Visualization     (Charting & Analysis)
└── snippets/          ← Wiederverwendbare UI-Components
    ├── charts.py         (Altair visualizations)
    └── dataframes.py     (Polars filtering)

examples/             ← Production-Ready Workflows
├── 01_interactive_data_profiler      (DuckDB + Polars)
├── 02_mlflow_experiment_console      (Experiment Tracking)
├── 03_pina_walrus_solver             (Physics Solver)
├── 04_hyperparameter_tuning          (AutoML)
├── 05_model_registry                 (MLflow Models)
├── 06_production_pipeline            (End-to-End)
└── 09_pina_live_monitoring           (Real-time)
```

## 🎯 AI-Human Cooperation Patterns

### Pattern 1: **Data Analysis Co-Pilot** ⭐⭐⭐ (HIGH PRIORITY)

**Basis**: `01_interactive_data_profiler.py`
- **Was es macht**: User lädt DuckDB Datenbank → wählt Table → filtert Spalten
- **AI Enhancement**: Claude analysiert Daten + schlägt intelligente Transformationen vor
- **Beispiel-Dialog**:
  ```
  User: "Lade die 'customers' Tabelle"
  Claude: "Ich sehe 10K rows mit 8 Spalten. Auffällungen:
           - customer_age: 18-78 Jahre
           - purchase_amount: stark rechtsschief
           → Sollen wir Outliers entfernen oder log-transformieren?"
  User: "Log-transformation"
  Claude: "Neue Spalte 'log_purchase' erstellt. Ergebnis:"
  ```

**Was zu tun ist**:
- [ ] Claude MCP Tool: `analyze_dataframe(df)` - Statistical insights
- [ ] Claude MCP Tool: `suggest_transformations(df)` - Recommendations
- [ ] Claude MCP Tool: `generate_eda_report(df)` - Auto-generated Report
- [ ] UI Widget: Transformation preview (before/after)

**Dependencies**: DuckDB, Polars, Pandas Profiling oder ähnlich

---

### Pattern 2: **Experiment Optimizer** ⭐⭐⭐ (HIGH PRIORITY)

**Basis**: `02_mlflow_experiment_console.py` + `04_hyperparameter_tuning.py`
- **Was es macht**: User schaut MLflow Experiment Results
- **AI Enhancement**: Claude schlägt nächste Hyperparameter vor + erklärt warum
- **Beispiel-Dialog**:
  ```
  User: "Letzte 5 Runs zeigen plateau bei 0.92 accuracy"
  Claude: "Ich sehe:
           - Learning rate: 1e-3 (konstant)
           - Batch size: 32 (konstant)
           - Epochs: 100 (erreicht plateau nach 60)

           Empfehlungen:
           1. Learning rate → 5e-4 (decay nach 50 epochs)
           2. Batch size → 64 (schneller Training)
           3. Early stopping bei 65 epochs

           Soll ich diese Hyperparams für neuen Run nutzen?"
  ```

**Was zu tun ist**:
- [ ] Claude MCP Tool: `analyze_mlflow_runs(experiment_id)` - Pattern recognition
- [ ] Claude MCP Tool: `suggest_hyperparams(runs_history)` - Intelligent suggestions
- [ ] Claude MCP Tool: `explain_run_difference(run1, run2)` - Comparative analysis
- [ ] Auto-run feature: "Start run with suggested params"

**Dependencies**: MLflow, Scikit-optimize oder Optuna Integration

---

### Pattern 3: **Physics Solver Co-Pilot** ⭐⭐⭐ (HIGH PRIORITY)

**Basis**: `03_pina_walrus_solver.py` + `09_pina_live_monitoring.py`
- **Was es macht**: User trainiert PINN für Physics-Probleme (Poisson, etc.)
- **AI Enhancement**: Claude hilft Problem formulieren + interpretiert Solver-Behavior
- **Beispiel-Dialog**:
  ```
  User: "Trainiere Poisson auf 2D domain"
  Claude: "Poisson-Gleichung erkannt: ∇²u = f
           Problem Konfiguration:
           - Domain: Unit Square ✓
           - Boundary conditions: Dirichlet ✓
           - Training loss: MSE ✓

           Hyperparameter Suggestions:
           - Learning rate: 1e-3 (für elliptic PDEs)
           - Architecture: [20,20,20,20,1] (balanced)
           - Epochs: 2000 (konvergiert typisch nach 1500)

           Starten?"
  User: "Ja"
  Claude: "Training läuft... (live loss plot)"
  Claude: "Nach 500 epochs: Loss = 0.023
           Konvergenz-Rate: -0.00004/epoch
           → Vorhersage: ≈1200 epochs bis Loss < 0.01"
  ```

**Was zu tun ist**:
- [ ] Claude MCP Tool: `analyze_pde_problem(problem)` - Domain expertise
- [ ] Claude MCP Tool: `predict_convergence(solver_state)` - Training ETA
- [ ] Claude MCP Tool: `validate_solution(solution)` - Physics-aware validation
- [ ] Live monitoring: Real-time "solver health" dashboard

**Dependencies**: PINA, Torch, Physics knowledge base

---

### Pattern 4: **Pipeline Builder** ⭐⭐ (MEDIUM PRIORITY)

**Basis**: `06_production_pipeline.py`
- **Was es macht**: Sequence of preprocessing → model → evaluation steps
- **AI Enhancement**: Claude generates optimal pipelines from requirements
- **Beispiel-Dialog**:
  ```
  User: "Build pipeline für time-series forecasting auf wind power data"
  Claude: "Zeit-Reihen Forecasting erkannt. Empfohlene Pipeline:

           1. Data Loading & Validation
              - Check missing values
              - Handle seasonality

           2. Feature Engineering
              - Lag features (t-1, t-7, t-30)
              - Rolling statistics
              - Fourier features für seasonality

           3. Preprocessing
              - StandardScaler (für LSTM)
              - Train/Test split (temporal)

           4. Modeling
              - LSTM (12 layer, 128 units)
              - Alternative: Transformer (for longer sequences)

           5. Evaluation
              - MAE, RMSE (prediction error)
              - MAPE (percentage error)

           Soll ich diese Pipeline generieren?"
  ```

**Was zu tun ist**:
- [ ] Claude MCP Tool: `generate_pipeline(task, data_info)` - Auto-pipeline
- [ ] Claude MCP Tool: `evaluate_pipeline_steps(pipeline)` - Quality metrics
- [ ] Pipeline templating system (with variations)

**Dependencies**: MLflow, Scikit-learn pipelines, Custom builders

---

## 🛠️ Technical Implementation Strategy

### Phase 1: Foundation (Week 1)
1. **Marimo MCP Integration** ✅ (schon gemacht)
   - Access_token management
   - Notebook CRUD operations

2. **Claude MCP Helpers**
   - Create standard utilities in `src/marimo_flow/mcp/`
   - Data analysis helpers
   - MLflow integration helpers

### Phase 2: Pattern 1 & 2 (Week 2-3)
1. Enhance `01_interactive_data_profiler.py` with Claude suggestions
2. Enhance `02_mlflow_experiment_console.py` + `04_hyperparameter_tuning.py`
3. Add interactive "try suggestion" buttons

### Phase 3: Pattern 3 & 4 (Week 4-5)
1. Enhance `03_pina_walrus_solver.py` with physics co-pilot
2. Enhance `06_production_pipeline.py` with auto-builder

### Phase 4: Documentation & Examples (Week 6)
1. Write comprehensive guides
2. Create tutorial notebooks
3. Package as proper examples

## 📁 New Folder Structure

```
src/marimo_flow/
├── core/              (existing - PINN stuff)
├── snippets/          (existing - UI components)
└── mcp/               (NEW - MCP helpers)
    ├── __init__.py
    ├── data_analysis.py      (Claude tools for data)
    ├── ml_optimization.py    (Claude tools for ML)
    ├── physics_helpers.py    (Claude tools for physics)
    └── pipeline_builders.py  (Claude tools for pipelines)

examples/
├── 01_interactive_data_profiler.py    (ENHANCE with AI)
├── 02_mlflow_experiment_console.py    (ENHANCE with AI)
├── 03_pina_walrus_solver.py           (ENHANCE with AI)
├── ai_human_notebooks/                (NEW)
│   ├── data_analyst_copilot.py        (Pattern 1)
│   ├── experiment_optimizer.py        (Pattern 2)
│   ├── physics_solver_copilot.py      (Pattern 3)
│   └── pipeline_builder.py            (Pattern 4)
└── tutorials/
    └── mcp_integration_guide.md       (Documentation)
```

## 🎓 Existing Code to Leverage

### Already Have:
- `ProblemManager` - Problem abstraction ✓
- `ModelFactory` - Model selection ✓
- `SolverManager` - Training management ✓
- `build_interactive_scatter` - Charting ✓
- `filter_dataframe` - Data manipulation ✓
- `build_optuna_history_figure` / `build_optuna_param_importance_figure` - Optuna visualization ✓

### Need to Add:
- Data profiling layer (for Pattern 1)
- Hyperparameter recommendation engine (for Pattern 2)
- Physics problem analyzer (for Pattern 3)
- Pipeline configuration DSL (for Pattern 4)

## 📊 Success Metrics

1. **Pattern 1 (Data Analysis)**: User can get Claude suggestions in 1 click
2. **Pattern 2 (Experiment)**: Hyperparameter suggestions improve accuracy by 5-10%
3. **Pattern 3 (Physics)**: Convergence time prediction ±20% accuracy
4. **Pattern 4 (Pipeline)**: Auto-generated pipelines match hand-tuned baselines

## 🚀 Quick Start for Implementation

```python
# In new src/marimo_flow/mcp/data_analysis.py

from typing import Any
import polars as pl

class DataAnalysisMCP:
    """Claude MCP tools for data analysis."""

    @staticmethod
    def analyze_dataframe(df: pl.DataFrame) -> dict[str, Any]:
        """Analyze DataFrame and return insights."""
        return {
            "shape": df.shape,
            "dtypes": dict(zip(df.columns, df.dtypes)),
            "nulls": df.null_count().to_dict(),
            "statistics": df.describe().to_dict(),
            "correlations": df.select(df.select(pl.numeric())).pearson_corr().to_dict(),
        }

    @staticmethod
    def suggest_transformations(df: pl.DataFrame) -> list[dict[str, str]]:
        """Suggest data transformations."""
        suggestions = []

        # Example: High cardinality → binning
        for col in df.columns:
            n_unique = df.select(col).n_unique()
            if n_unique > 100:
                suggestions.append({
                    "column": col,
                    "issue": "High cardinality",
                    "suggestion": "Binning or target encoding",
                })

        return suggestions
```

Das ist die Grundlage - wir bauen darauf auf!
