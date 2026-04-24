---
name: pina-3d
description: Guidance for composing 3D PINA problems — sampling budgets, activation choices, and gotchas specific to three spatial axes plus optional time.
triggers:
  - 3d problem
  - three dimensional pde
  - volumetric pde
  - 3d poisson
  - 3d heat
  - 3d navier-stokes
  - lid-driven cavity
  - 3d wave
---

# PINA 3D Problems

3D is the engineering default for this project. The composer already
handles arbitrary axis sets — ``{"x": [...], "y": [...], "z": [...]}``
produces a 3D ``SpatialProblem`` with zero Python changes; adding
``"t": [...]`` promotes to ``TimeDependentProblem``. This skill
captures the **scaling + numerical** considerations that only become
important once you leave 2D.

## Composition — no surprises

```python
spec = ProblemSpec(
    name="heat_3d",
    output_variables=["u"],
    domain_bounds={
        "x": [0.0, 1.0], "y": [0.0, 1.0], "z": [0.0, 1.0],
        "t": [0.0, 1.0],
    },
    subdomains=[
        SubdomainSpec(name="D", bounds={
            "x": [0.0, 1.0], "y": [0.0, 1.0], "z": [0.0, 1.0], "t": [0.0, 1.0],
        }),
        # Faces: pin exactly one axis
        SubdomainSpec(name="x0", bounds={
            "x": 0.0, "y": [0,1], "z": [0,1], "t": [0,1]
        }),
        # … and so on for the other five walls.
    ],
    equations=[
        EquationSpec(
            name="heat",
            form="u_t - alpha*(u_xx + u_yy + u_zz)",
            outputs=["u"],
            derivatives=[
                DerivativeSpec(name="u_t",  field="u", wrt=["t"]),
                DerivativeSpec(name="u_xx", field="u", wrt=["x","x"]),
                DerivativeSpec(name="u_yy", field="u", wrt=["y","y"]),
                DerivativeSpec(name="u_zz", field="u", wrt=["z","z"]),
            ],
            parameters={"alpha": 0.1},
        ),
    ],
    conditions=[
        ConditionSpec(subdomain="D", kind="equation", equation_name="heat"),
        # Dirichlet walls via fixed_value; IC via a t=0 subdomain + equation_inline.
    ],
)
```

## Sampling budgets

A 2D Poisson trains well on ~1k collocation points; 3D needs **10x**
that just to cover the volume evenly. Rule of thumb:

- 3D steady-state: ``n_points = 8000–16000``.
- 3D + time: ``n_points = 20000–40000``.
- Use ``sample_mode="latin"`` — uniform random gets clustery quickly
  in 3D.

## Architecture tuning

- Minimum hidden width: **64**. The 8×8 FeedForward that trains 1D
  Burgers in 2 epochs will not fit 3D without adjustment.
- Depth: 5–8 layers; deeper helps higher-frequency features but Adam
  stalls earlier, use LBFGS for the last 20% of training.
- Activations: ``tanh`` or ``silu``. Avoid ``relu`` — its second
  derivative is zero, which nulls the viscous / diffusive terms.
- For Navier-Stokes 3D, **Fourier Feature Nets** (a.k.a. RFF) beat
  plain FeedForward. ``ModelManager.create("fno", ...)`` is the right
  first pick.

## Visualisation

The provenance dashboard (Phase C3) uses pyvista for 3D rendering.
Export inference output via ``Trainer.test()`` on a volumetric grid
(e.g. 32³) and load the resulting numpy array into a ``pyvista.ImageData``.
Until that's wired, fall back to 2D slices — the 2D viz in
``examples/02_provenance_dashboard.py`` takes ``(x, y, value)`` at
fixed z, t.

## Gotchas

- PINA's ``input_variables`` on a 3D+time problem is ``["x","y","z","t"]``
  — verify with ``cls().input_variables`` after composing.
- Derivative operators scale with the number of axes, so the
  residual cost grows as **O(d²)** for a full Laplacian. Do not
  compose a 3D problem with ``Δu + …`` as a single ``wrt=["x","x"]``;
  list each direction as its own derivative so the operator graph
  stays shallow.
- Memory: one 40k × (3+1+hidden_width) float32 tensor is ~13 MB per
  batch. Watch GPU memory if you push past 100k points.
