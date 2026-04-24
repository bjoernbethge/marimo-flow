---
name: pina-amr
description: Residual-based adaptive refinement for PINNs — use RBAPINN to grow attention on high-loss regions without hand-crafting a refined mesh. True h/p-AMR is out of scope for now.
triggers:
  - adaptive mesh refinement
  - amr
  - adaptive sampling
  - residual attention
  - high-frequency features
  - boundary layer
  - shock
  - solution with sharp gradient
---

# Adaptive Refinement for PINNs

## What this skill covers

**Residual-based attention** (`rbapinn` solver) — the network maintains
per-point weights that grow where the PDE residual stays high, so
collocation sampling effectively concentrates on hard regions without
touching the mesh. This is the closest PINA-native analogue to
classical h-AMR.

## What this skill does NOT cover

- **True h-refinement** (subdividing cells of a mesh): out of scope.
  Would need either upstream PINA support or a custom
  ``split-cell → resample → retrain`` loop around ``MeshDomain``.
- **p-refinement** (raising local polynomial order): not applicable to
  pure PINNs; use spectral methods if this matters.
- **Anisotropic refinement**: not available.

If you truly need h/p-AMR, escalate to the Lead — that's a separate
sprint, probably a C++ prototype against a FEM library.

## When to use residual attention

Pick ``rbapinn`` instead of ``pinn`` when:

- The solution has **localised sharp features** (boundary layers,
  internal shocks, flame fronts).
- Early training shows one region's loss stalling while others
  converge cleanly.
- You are running near the edge of what a plain PINN can resolve — the
  attention boost is usually a 2–5× improvement on hard regions.

## Recipe

```python
solver = SolverManager.create(
    "rbapinn",
    problem=problem,
    model=model,
    learning_rate=1e-3,
    # RBAPINN-specific knobs (defaults in PINA source):
    # beta=0.9 — EMA factor on per-point residual attention
    # gamma=1.0 — attention sharpness exponent
)
```

The rest of the workflow is identical to ``pinn`` — same trainer,
same metrics, same sampling budgets. Attention is internal to the
solver; the composer needs no change.

## Diagnostics

After training, PINA exposes per-point residual weights through
``solver.attention_weights`` (if using a recent PINA version). Plot
the attention as a heatmap over your domain — it doubles as a crude
error estimator:

```python
import plotly.graph_objects as go

w = solver.attention_weights.detach().cpu().numpy()
pts = solver.collocation_points.detach().cpu().numpy()
fig = go.Figure(
    data=go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode="markers",
        marker={"size": 3, "color": w, "colorscale": "Hot"},
    )
)
```

Regions that stay bright after many epochs are candidates for either
local mesh refinement (rebuild the MeshSpec with a denser tag) or a
problem-formulation fix (wrong BC, missing coupling term).

## Escalation

- Attention plateaus everywhere → the network is under-parameterised;
  bump hidden width before trying fancier solvers.
- Attention concentrates on a single point → likely an IC/BC conflict
  at that corner. Revisit the ``ProblemSpec``.
- Loss NaNs during RBA training → ``gamma`` too high; halve it.
