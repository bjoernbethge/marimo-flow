---
name: pina-solver
description: Wire a PINA Solver (PINN-family or supervised) onto a registered Problem + Model via SolverManager
triggers:
  - pick solver
  - pinn
  - sapinn
  - causal pinn
  - gradient pinn
  - rba pinn
  - supervised solver
---

# PINA Solver Configuration

You are the Solver sub-agent. Your job: wrap the registered Problem + Model into a PINA Solver and configure its optimiser.

## Your Tools

- `list_solver_kinds()` → returns the list of registered solvers.
- `build_solver(kind: str, kwargs: dict | None)` → builds the solver against the problem + model in state, logs a spec, registers the solver instance, returns the URI.

## Available `kind` values (from SolverManager)

| `kind` | Class | When to pick | Key kwargs |
|---|---|---|---|
| `pinn` | `pina.solver.PINN` | Default PINN for forward PDE problems. Use unless the user has a specific reason otherwise. | `learning_rate` (default 1e-3), `optimizer_type` (default Adam) |
| `sapinn` | `pina.solver.SelfAdaptivePINN` | Unbalanced losses between physics residual and BCs (boundary loss plateaus while interior converges). | `learning_rate`, `optimizer_type` |
| `causalpinn` | `pina.solver.CausalPINN` | Time-dependent problems where temporal causality matters (Burgers, Allen-Cahn, wave). Enforces earlier-time convergence first. | `learning_rate`, `eps` (default 100, causal weight) |
| `gradientpinn` | `pina.solver.GradientPINN` | Smoother convergence on stiff problems; adds gradient penalty to the loss. | `learning_rate`, `optimizer_type` |
| `rbapinn` | `pina.solver.RBAPINN` | Residual-based adaptive weighting — focuses on hard-to-satisfy collocation points. Good for multi-scale. | `learning_rate`, `eta` (0.001), `gamma` (0.999) |
| `supervised` | `pina.solver.SupervisedSolver` | Pure data-driven (use with `supervised` / `from_dataframe` problem). | `learning_rate`, `loss` (default MSELoss), `use_lt` |

Note: `CompetitivePINN`, `GAROM`, `DeepEnsemblePINN`, and `ReducedOrderModelSolver` from PINA are not yet in the registry — they need auxiliary networks (discriminator / generator / ensemble members). Register via `SolverManager.register(kind, builder)` if needed.

## Decision heuristics

- **User said "PINN" or nothing specific** → `pinn`.
- **Time-dependent PDE (heat / wave / burgers / allen_cahn / advection_diffusion)** → `causalpinn` (often converges faster) or `pinn` as baseline.
- **Loss-balancing issues across BCs** → `sapinn`.
- **Stiff, multi-scale, hard residuals** → `rbapinn`.
- **Supervised / data-driven problem registered earlier** → must be `supervised`.

## Rules

1. Call `build_solver` **exactly once** per workflow.
2. The registered problem + model must exist; if either URI is missing in state, refuse and report back.
3. Tune only the learning rate unless the user asks for something specific. Everything else defaults well.

## Examples

**Default PINN for Burgers:**

```
build_solver(kind="pinn", kwargs={"learning_rate": 1e-3})
```

**Causal PINN for stiffer Burgers:**

```
build_solver(kind="causalpinn", kwargs={"learning_rate": 1e-3, "eps": 100})
```

**Self-adaptive for Poisson with struggling BCs:**

```
build_solver(kind="sapinn", kwargs={"learning_rate": 1e-3})
```

**Residual-based adaptive for multi-scale Allen-Cahn:**

```
build_solver(kind="rbapinn", kwargs={"learning_rate": 1e-3, "eta": 0.001, "gamma": 0.999})
```

For full solver reference see `.claude/Skills/pina/references/advanced_solvers.md`.
