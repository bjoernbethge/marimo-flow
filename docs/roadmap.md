# Roadmap: beyond the current PINA presets

Last updated: 2026-04-24.

**Status**: Phase A-0 complete and realigned as a composition-first
catalog (no more hardcoded `ProblemKind`). The earlier plan for Phase A
(Navier-Stokes / Maxwell / Elasticity as new hardcoded factories) has
been dropped — the **composer** (`services/composer.py::compose_problem`)
reaches those PDEs as soon as an agent writes a suitable `ProblemSpec`.
No Python changes required per new PDE family.

## Architecture pivot (2026-04-24)

The initial Phase A-0 tried to layer a persistent preset catalog on top
of the existing hardcoded `ProblemManager.create_*` factories. That
missed the point: agents could tweak parameters but could not invent
new PDE families. The user flagged it; the pivot is:

* **`services/composer.py::compose_problem`** accepts a full typed
  `ProblemSpec` (`EquationSpec` + `SubdomainSpec` + `ConditionSpec`)
  and compiles it via `sympy.lambdify` + `pina.operator.grad/laplacian`
  into a `pina.Problem` subclass at runtime.
* **No `ProblemKind` literal.** Any PDE that sympy can express and
  PINA's operators can differentiate is reachable.
* **The catalog stores compositions**, not parameter bundles. Agents
  register successful `ProblemSpec` values and clone them with
  overrides; no builtin seeding.
* **Model and solver layers keep their `Literal` kinds** (`ModelKind`,
  `SolverKind`) because those *are* finite choice spaces over PINA's
  built-in neural architectures and solver algorithms. An agent picking
  between `feedforward` / `fno` / `deeponet` is legitimate; an agent
  picking between `burgers` and `heat` is not (it should be building a
  `ProblemSpec` that encodes the physics).

Baseline after the pivot: **180 tests pass**, Burgers 1D composes and
trains end-to-end via the composer (no hardcoded preset).

Background context: [`CLAUDE.md`](../CLAUDE.md), [`README.md`](../README.md).

## Leitplanken

- **3D is the default**. Engineering problems are rarely 2D. The
  composer handles arbitrary axis sets in `domain_bounds` —
  `{"x": [...], "y": [...], "z": [...]}` makes a 3D problem, add
  `"t": [...]` for 3D+time. 2D stays as a reduction.
- **Observations come from the Data agent, never from the user.**
  `TaskSpec.observables` (what to measure) is triage-extracted.
  `ObservationSpec` (concrete `(x,y,z,t,value)` tuples) is loaded from
  file or synthesised by the Data agent — the user never hand-codes
  numpy arrays.
- **No new dependencies without a recon step**. Each phase starts with
  a recon task against the current PINA release.
- **All new deps via `uv add`**, never ad-hoc `pip install`.

## Overview

| Phase | Theme | Effort | External deps | Depends on |
|---|---|---|---|---|
| **A-0** ✅ | Preset catalog + composer + curator_toolset | — | `sympy` | — |
| **B** | Inverse path + multiphysics + Data agent | M | — | A-0 |
| **C** | Mesh/CAD geometry + 3D visualisation | L | `meshio`, `pyvista` (opt.) | — |
| **C2** | AMR (separate milestone) | XL | PINA upstream or custom | C |
| **D** | PDE-constrained optimisation | M | `cvxpy` optional | A-0, B |
| **E** | Stochastic + non-local PDEs | M | — | — |
| **F** | Real-time control / MPC | L | `do-mpc` or `cvxpy` | D |

Effort legend: S = 1–2 sessions, M = 2–3, L = 3–4, XL = 5+.

---

## Phase A-0 — DONE

Composer-first catalog. What's shipped:

- `schemas/equation.py` — `EquationSpec`, `DerivativeSpec`, `SubdomainSpec`, `ConditionSpec`.
- `schemas/problem.py` — composition-only `ProblemSpec` (no kind enum).
- `services/composer.py` — `compose_problem(spec)` + `build_equation(spec)`.
- `services/preset_catalog.py` — DuckDB-backed store for user-authored compositions + optional YAML mirror.
- `toolsets/problem.py` — `compose_problem`, `inspect_problem`, `list_input_vars_hint`.
- `toolsets/curator.py` — `search_presets`, `describe_preset`, `register_preset`, `clone_preset`, `deprecate_preset`.
- Smoke test: 1D viscous Burgers composes + trains 2 epochs with a 8×8 FeedForward on CPU.

---

## Phase B — Inverse problems, multiphysics, Data agent

Let agents handle parameter identification and coupled physics through
the same composer, plus automate observation loading.

| # | Title | Effort |
|---|---|---|
| 12 | Recon: PINA inverse-problem pattern (LearnableParameter) | S |
| 13 | Schemas: `ObservationSpec` + `UnknownParameterSpec` | M |
| 30 | **Data agent: auto-ingest observations from file + synthetic generator** | M |
| 14 | Composer support for inverse problems | M |
| 15 | Multiphysics via multiple EquationSpecs on one ProblemSpec | M |
| 16 | Skills: `pina-inverse`, `pina-multiphysics`, `pina-3d` | M |

**Technical notes**

- `ObservationSpec.source ∈ {data_file, synthetic, live_sensor}`. Data
  agent dispatches:
  - `data_file` → `load_observations_from_file(path, field_name, format)`
    parses CSV / Parquet / NPZ and recognises `(x,y,z,t,field_value)`.
  - `synthetic` → `generate_synthetic_observations(n_points, true_parameters, noise_sigma)`
    runs a forward solve with the true parameters, samples n points,
    adds Gaussian noise.
  - `live_sensor` → subscribes to MQTT / Kafka (Phase F).
- `UnknownParameterSpec(name, initial, bounds, trainable)` becomes a
  `pina.LearnableParameter` inside the compiled torch callable.
- Multiphysics works without new schema: two EquationSpecs that share
  `output_variables` and both point at the interior subdomain produce
  two independent loss terms. Reference example: thermo-elasticity
  (heat + elasticity + thermal-strain coupling term).
- DuckDB extensions: tables `observations`, `unknown_parameters`.

---

## Phase C — 3D geometry, mesh, CAD, visualisation

Move beyond CartesianDomain. Mesh import (STL/OBJ/VTK/GMSH), optional
CAD bridge (STEP/IGES), interactive 3D rendering in the dashboard.

| # | Title | Effort |
|---|---|---|
| 17 | Recon: non-cartesian PINA domains + mesh integration | S |
| 18 | `uv add meshio` + `MeshSpec` schema + `meshes` table | M |
| 19 | `MeshDomain` adapter for the composer | L |
| 20 | `pina-geometry` skill + CAD bridge via pyvista/OCCT | M |
| 31 | 3D visualisation in the provenance dashboard (pyvista) | M |
| 32 | Demo notebook: 3D Navier-Stokes lid-driven cavity | S |

**Separate milestone**

| # | Title | Effort |
|---|---|---|
| 21 | **AMR** (adaptive mesh refinement) | XL |

**Technical notes**

- `MeshSpec(path, format, units, tags: dict[str, list[int]])` — tags
  map cell groups to physical IDs (for BCs per face).
- `MeshDomain` implements PINA's Domain interface: point-sampling via
  barycentric coords per cell, BC-subdomain lookup via mesh tags.
- `SubdomainSpec` gains an optional `mesh_ref` escape hatch so
  conditions can target mesh tags instead of cartesian bounds.
- CAD: `pyvista` (STL/VTK/GLTF native) + optional `pythonOCC` for
  STEP/IGES. Both soft-deps via `uv add --optional`.
- AMR: `RBAPINN` already covers adaptive sampling. Real h/p-AMR needs
  either PINA upstream support or a custom refinement loop.

---

## Phase D — PDE-constrained optimisation

Design loop over PINN composers — topology, shape, material optimisation
with physical constraints. Binds to the existing Optuna stack in
`core/visualization.py`.

| # | Title | Effort |
|---|---|---|
| 22 | Design agent + `OptimizationPlan` + `DesignVariableSpec` | M |
| 23 | `design_toolset` with penalty / augmented-Lagrangian handler | M |

**Technical notes**

- New agent role `design`.
- `OptimizationPlan(objective, design_variables, constraints, method)`
  with `method ∈ {optuna_tpe, scipy_slsqp, penalty, augmented_lagrangian}`.
- Constraint handler: penalty (`loss_total = loss_pde + λ·max(0, g(x))²`)
  vs. augmented-Lagrangian (λ updates per outer loop).
- Smoke-test: 2D/3D topology optimisation of a plate under load
  (minimal material at a stiffness constraint).

---

## Phase E — Stochastic + non-local PDEs

| # | Title | Effort |
|---|---|---|
| 24 | Stochastic PDEs: `NoiseSpec` + SDE solver variant | M |
| 25 | Non-local / fractional PDEs | M |

**Technical notes**

- `NoiseSpec(type="white|colored|fbm", intensity, correlation)` extends
  the composer to wrap an EquationSpec with an additive noise term.
  Solver: Monte-Carlo over noise realisations with a mean PINN loss, or
  variational PINN.
- Fractional Laplace as a custom `pina.Equation` with spectral or
  diffusive-representation quadrature. Expose via a new derivative kind
  in the composer. Test against an analytical fractional-Poisson
  solution on an interval.

---

## Phase F — Real-time control / MPC

Use a trained PINN as a dynamics model inside an MPC loop — closed-loop
control driven by a physics-informed surrogate.

| # | Title | Effort |
|---|---|---|
| 26 | Recon: MPC libraries (do-mpc / cvxpy / acados) | M |
| 27 | `control/` module + `ControlAgent` + `ControlPlan` schema | L |
| 28 | Closed-loop demo + `control_dashboard` | M |

**Technical notes**

- `src/marimo_flow/control/` parallel to `core/` and `agents/` — keeps
  the control domain from leaking into the PINN kernel.
- `ControlPlan(horizon, objective, constraints, surrogate_uri)`. The
  `surrogate_uri` points at a trained solver in the DuckDB `artifacts`
  table.
- Default library: `do-mpc` (high-level, CasADi-based). `cvxpy` only
  where the problem is convex. `acados` (C++) is out of scope for this
  repo — sibling project material.
- Smoke-test: 1D heat-rod stabilisation or inverted pendulum with a
  PINN surrogate as the `dynamics_function`.

---

## Suggested sequence

1. **Phase B** next — builds on the composer and adds inverse +
   multiphysics with minimal new machinery. Task #30 (Data agent) is
   the keystone so observables are truly agent-managed.
2. **Phase C** in parallel or after — orthogonal to B. C2 (AMR) stays a
   separate sprint.
3. **Phase D** needs B (design-over-problem).
4. **Phase E** orthogonal, any time.
5. **Phase F** last — needs trained surrogates from A–D as the
   dynamics model.

## Non-goals

- **Shock-fitting / Riemann solvers** for hyperbolic conservation laws.
  PINN is structurally poor at this; use FV/DG methods (dolfinx,
  Clawpack, OpenFOAM) in a separate project.
- **Volumetric rendering of tera-scale fields**. pyvista is fine for
  lab scale; Paraview / Catalyst stays outside.
- **Real HPC parallelism**. Lightning covers DDP; SLURM / Horovod go
  out of scope.

## Open questions before kick-off

- Phase C: pyvista vs. trimesh for point-sampling on meshes? Task #17
  decides.
- Phase D: own Optuna-objective wrapper vs. skopt vs. BoTorch? Recon
  before Task #22.
- Phase F: `control/` as a sibling module vs. a separate repo? The
  scope test in Task #26 will tell.

---

All 20 open tasks live under `TaskList`. Phase A-0 complete; Phase B is
the next natural step.
