---
name: pina-inverse
description: Compose inverse / parameter-identification PINA problems — declare UnknownParameterSpec, attach ObservationSpec from data or synthetic sampling, and the composer wires PINA InverseProblem automatically.
triggers:
  - inverse problem
  - parameter identification
  - parameter estimation
  - data assimilation
  - calibrate pde
  - fit viscosity
  - infer coefficient
  - learnable parameter
---

# PINA Inverse Problems

Use this skill when the goal is **to learn one or more scalar PDE
coefficients from observed data** instead of solving the forward
problem for a known PDE. The composer reuses all the same primitives
— you only add two things: `UnknownParameterSpec` (what to learn) and
`ObservationSpec` (the data to fit).

## When it applies

- "Given these ``(x, t, u)`` measurements, what is the viscosity?"
- "Estimate the diffusion coefficient from temperature sensors."
- "Calibrate the source term location from boundary observations."

The forward PDE is still needed — you describe its structure exactly
as in `pina-problem`. The unknowns then appear as bare symbols inside
`EquationSpec.form` **without** being listed in `EquationSpec.parameters`.
The composer detects them and routes them through PINA's
`params_` dict so backprop treats them as learnable tensors.

## Contract

1. Build a `ProblemSpec` exactly like for a forward problem (equations,
   subdomains, conditions).
2. Add `unknowns: list[UnknownParameterSpec]` — each entry gets a
   `(low, high)` prior range used for uniform initialisation.
3. Add `observations: list[ObservationSpec]` with materialised
   ``points`` / ``values``. If the user did **not** provide raw data,
   hand the task to the Data agent:
   - ``load_observations_from_file(path, field, axes)`` for CSVs,
     Parquet, NPZ.
   - ``generate_synthetic_observations(truth_form, axes, field,
     axis_bounds, n_points, noise_sigma, true_parameters)`` if you need
     a benchmarking reference.

Never paste raw numpy arrays into the ProblemSpec yourself — the Data
agent is the single owner of observations.

## Worked example — infer viscosity from 1D Burgers

```python
from marimo_flow.agents.schemas import (
    ConditionSpec, DerivativeSpec, EquationSpec, ObservationSpec,
    ProblemSpec, SubdomainSpec, UnknownParameterSpec,
)

# 1. Ask Data agent for observations (here: synthetic with true ν=0.01).
obs_dict = generate_synthetic_observations(
    truth_form="-sin(pi*x)",   # initial profile; placeholder solution
    axes=["x", "t"], field="u",
    axis_bounds={"x": [-1.0, 1.0], "t": [0.0, 1.0]},
    n_points=200, noise_sigma=0.01,
    true_parameters={"pi": 3.141592653589793},
)

spec = ProblemSpec(
    name="burgers_inverse",
    output_variables=["u"],
    domain_bounds={"x": [-1.0, 1.0], "t": [0.0, 1.0]},
    subdomains=[
        SubdomainSpec(name="left",  bounds={"x": -1.0, "t": [0.0, 1.0]}),
        SubdomainSpec(name="right", bounds={"x":  1.0, "t": [0.0, 1.0]}),
        SubdomainSpec(name="D",     bounds={"x": [-1.0, 1.0], "t": [0.0, 1.0]}),
    ],
    equations=[
        EquationSpec(
            name="burgers",
            form="u_t + u*u_x - nu*u_xx",  # nu is the UNKNOWN
            outputs=["u"],
            derivatives=[
                DerivativeSpec(name="u_t",  field="u", wrt=["t"]),
                DerivativeSpec(name="u_x",  field="u", wrt=["x"]),
                DerivativeSpec(name="u_xx", field="u", wrt=["x","x"]),
            ],
            # parameters is empty: nu is inferred via params_.
        ),
    ],
    conditions=[
        ConditionSpec(subdomain="left",  kind="fixed_value", value=0.0),
        ConditionSpec(subdomain="right", kind="fixed_value", value=0.0),
        ConditionSpec(subdomain="D",     kind="equation", equation_name="burgers"),
    ],
    unknowns=[UnknownParameterSpec(name="nu", low=0.001, high=0.1)],
    observations=[ObservationSpec(**obs_dict)],
)
cls = compose_problem(spec)          # emits SpatialProblem + TimeDependentProblem + InverseProblem
```

## What the composer does differently

- Adds `pina.problem.InverseProblem` to the base tuple and fills
  `unknown_parameter_domain` from the declared bounds.
- Rewrites the residual with a 3-arg signature
  ``(input_, output_, params_)``; PINA auto-detects this and treats
  the problem as inverse.
- Emits one extra `Condition(input=…, target=…)` per observation.

## Validation

- Inspect the compiled problem with ``inspect_problem``; the report
  lists ``unknown_variables`` and each observation's point count.
- After training, read the learned parameter values from
  ``solver.problem.unknown_parameters`` — those are
  ``torch.nn.Parameter`` tensors.
- Typical gotcha: if the user gave a single observation row, the
  fitter collapses to a single scalar constraint and the unknown stays
  at its initial uniform draw. Reject observations below ~20 points.

## Escalation

If training fails to reduce the observation loss below 10× the noise
floor, escalate to the Lead — the problem may be ill-posed (sensors
too correlated, multiple unknowns with similar effect, or a data/PDE
mismatch).
