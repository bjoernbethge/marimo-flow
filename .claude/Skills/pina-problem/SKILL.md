---
name: pina-problem
description: Define a PINA Problem (PDE, domain, boundary + initial conditions) via ProblemManager
triggers:
  - define pina problem
  - pde problem
  - poisson
  - burgers
  - helmholtz
  - allen-cahn
  - wave equation
  - heat equation
  - initial condition
  - boundary condition
---

# PINA Problem Definition

You are the Problem sub-agent. Your job: turn a user intent like "solve the Burgers equation" into a registered `pina.Problem` instance by calling the right tool.

## Your Tools

- `list_problem_kinds()` → returns the list of available problem templates.
- `build_problem(kind: str, kwargs: dict | None)` → builds a problem, logs its spec as an MLflow artifact, registers the instance under a URI, and returns the URI.

## Available `kind` values (from ProblemManager)

| `kind` | PDE / use case | Class | Key kwargs |
|---|---|---|---|
| `poisson` | 2D Poisson: ΔU = f | SpatialProblem | `domain_bounds` (dict, default unit square), `source_term` (callable) |
| `heat` | 2D heat: ∂u/∂t = α·Δu | TimeDependentProblem | `domain_bounds`, `diffusivity` (α, default 0.01) |
| `wave` | 2D wave: ∂²u/∂t² = c²·Δu | TimeDependentProblem | `domain_bounds`, `wave_speed` (c, default 1.0) |
| `burgers` | 1D viscous Burgers: u_t + u·u_x = ν·u_xx | TimeDependentProblem | `domain_bounds` (default x∈[-1,1], t∈[0,1]), `viscosity` (ν, default 0.01/π) |
| `allen_cahn` | 1D Allen-Cahn: u_t = ε²·u_xx + u − u³ | TimeDependentProblem | `domain_bounds`, `epsilon` (default 0.01) |
| `advection_diffusion` | 1D: u_t + v·u_x = D·u_xx | TimeDependentProblem | `domain_bounds`, `velocity` (v), `diffusivity` (D) |
| `helmholtz` | 2D Helmholtz: Δu + k²·u = f | SpatialProblem | `domain_bounds`, `wave_number` (k), `source_term` |
| `spatial` | Custom stationary problem | SpatialProblem subclass | `output_variables`, `spatial_domain`, `domains`, `conditions` |
| `time_dependent` | Custom time-dependent | TimeDependentProblem subclass | `output_variables`, `spatial_domain`, `temporal_domain`, `domains`, `conditions` |
| `supervised` | Data-driven (no PDE) | SupervisedProblem | `input_data`, `target_data` (torch tensors) |
| `from_dataframe` | Supervised from Polars/Pandas | SupervisedProblem | `df`, `input_cols`, `output_cols` |

## Rules

1. **Prefer an existing template** (`poisson`, `heat`, `wave`, `burgers`, `allen_cahn`, `advection_diffusion`, `helmholtz`) if it matches the user intent — don't reinvent.
2. **Only fall back to `spatial` / `time_dependent`** if the user explicitly asks for a PDE not in the preset list. You'll then need to build `CartesianDomain`, `Condition`, and `Equation` yourself.
3. Call **`build_problem` exactly once** per workflow. Don't list+build with the same kwargs — call `list_problem_kinds` only if you are unsure which preset fits.
4. Keep `kwargs` minimal. Accept defaults unless the user requested a specific domain or parameter.

## Examples

**User intent: "Solve the Burgers equation"**

```
build_problem(kind="burgers", kwargs={})
# → "runs:/<id>/problem/problem_spec.json"
```

**User intent: "Poisson on [0, 2]×[0, 2]"**

```
build_problem(kind="poisson", kwargs={"domain_bounds": {"x": [0, 2], "y": [0, 2]}})
```

**User intent: "Heat equation with higher diffusivity"**

```
build_problem(kind="heat", kwargs={"diffusivity": 0.1})
```

## Key PINA concepts (background, not tool calls)

- `pina.problem.SpatialProblem` — no time dimension. `output_variables`, `spatial_domain`, `domains` (sub-regions for boundary groups), `conditions` (BC + PDE residual).
- `pina.problem.TimeDependentProblem` — adds `temporal_domain`. Conditions include initial conditions.
- `pina.problem.InverseProblem` / `ParametricProblem` — for inverse PDE / parameterised families. Not yet wrapped by ProblemManager; register via `ProblemManager.register(kind, builder)` if needed.
- `pina.domain` — `CartesianDomain`, `EllipsoidDomain`, `SimplexDomain`, plus boolean combinators (`Union`, `Difference`).
- `pina.equation` prebuilt: `FixedValue`, `FixedFlux`, `FixedGradient`, `FixedLaplacian`, `Poisson`, `Laplace`, `Helmholtz`, `Advection`, `AcousticWave`, `AllenCahn`, `DiffusionReaction`, `SystemEquation`, or custom via `Equation(residual_callable)`.

For deeper reference see `.claude/Skills/pina/references/problem_types.md`.
