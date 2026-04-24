"""Tests for the PDE composer (compose_problem + build_equation).

Uses real sympy + PINA operators — no mocks. Verifies that composed
problems have the right structure (output_vars, subdomains, time-dependency)
and that the symbolic residual produces a torch tensor with the expected
autograd signal.
"""

from __future__ import annotations

import pytest
import torch
from pina.problem import SpatialProblem, TimeDependentProblem

from marimo_flow.agents.schemas import (
    ConditionSpec,
    DerivativeSpec,
    EquationSpec,
    ProblemSpec,
    SubdomainSpec,
)
from marimo_flow.agents.services.composer import build_equation, compose_problem


def _burgers_spec(name: str = "burgers_1d") -> ProblemSpec:
    return ProblemSpec(
        name=name,
        output_variables=["u"],
        domain_bounds={"x": [-1.0, 1.0], "t": [0.0, 1.0]},
        subdomains=[
            SubdomainSpec(name="left", bounds={"x": -1.0, "t": [0.0, 1.0]}),
            SubdomainSpec(name="right", bounds={"x": 1.0, "t": [0.0, 1.0]}),
            SubdomainSpec(name="t0", bounds={"x": [-1.0, 1.0], "t": 0.0}),
            SubdomainSpec(name="D", bounds={"x": [-1.0, 1.0], "t": [0.0, 1.0]}),
        ],
        equations=[
            EquationSpec(
                name="burgers",
                form="u_t + u*u_x - nu*u_xx",
                outputs=["u"],
                derivatives=[
                    DerivativeSpec(name="u_t", field="u", wrt=["t"]),
                    DerivativeSpec(name="u_x", field="u", wrt=["x"]),
                    DerivativeSpec(name="u_xx", field="u", wrt=["x", "x"]),
                ],
                parameters={"nu": 0.01},
            ),
            EquationSpec(
                name="ic",
                form="u + sin(pi*x)",
                outputs=["u"],
                derivatives=[],
                parameters={"pi": 3.141592653589793},
            ),
        ],
        conditions=[
            ConditionSpec(subdomain="left", kind="fixed_value", value=0.0),
            ConditionSpec(subdomain="right", kind="fixed_value", value=0.0),
            ConditionSpec(subdomain="t0", kind="equation", equation_name="ic"),
            ConditionSpec(subdomain="D", kind="equation", equation_name="burgers"),
        ],
    )


def test_compose_burgers_is_time_dependent():
    cls = compose_problem(_burgers_spec())
    assert issubclass(cls, TimeDependentProblem)
    assert issubclass(cls, SpatialProblem)
    assert cls.__name__ == "burgers_1d"


def test_compose_stationary_poisson_2d():
    spec = ProblemSpec(
        name="poisson_2d",
        output_variables=["u"],
        domain_bounds={"x": [0.0, 1.0], "y": [0.0, 1.0]},
        subdomains=[
            SubdomainSpec(name="D", bounds={"x": [0.0, 1.0], "y": [0.0, 1.0]}),
        ],
        equations=[
            EquationSpec(
                name="poisson",
                form="u_xx + u_yy + sin(pi*x)*sin(pi*y)",
                outputs=["u"],
                derivatives=[
                    DerivativeSpec(name="u_xx", field="u", wrt=["x", "x"]),
                    DerivativeSpec(name="u_yy", field="u", wrt=["y", "y"]),
                ],
                parameters={"pi": 3.141592653589793},
            ),
        ],
        conditions=[
            ConditionSpec(subdomain="D", kind="equation", equation_name="poisson"),
        ],
    )
    cls = compose_problem(spec)
    assert issubclass(cls, SpatialProblem)
    assert not issubclass(cls, TimeDependentProblem)
    assert "D" in cls.domains


def test_compose_poisson_3d_unit_cube():
    """3D problems work without any code changes — just more axes."""
    spec = ProblemSpec(
        name="poisson_3d",
        output_variables=["u"],
        domain_bounds={
            "x": [0.0, 1.0],
            "y": [0.0, 1.0],
            "z": [0.0, 1.0],
        },
        subdomains=[
            SubdomainSpec(
                name="D",
                bounds={"x": [0.0, 1.0], "y": [0.0, 1.0], "z": [0.0, 1.0]},
            ),
        ],
        equations=[
            EquationSpec(
                name="poisson3d",
                form="u_xx + u_yy + u_zz",
                outputs=["u"],
                derivatives=[
                    DerivativeSpec(name="u_xx", field="u", wrt=["x", "x"]),
                    DerivativeSpec(name="u_yy", field="u", wrt=["y", "y"]),
                    DerivativeSpec(name="u_zz", field="u", wrt=["z", "z"]),
                ],
            ),
        ],
        conditions=[
            ConditionSpec(
                subdomain="D", kind="equation", equation_name="poisson3d"
            ),
        ],
    )
    cls = compose_problem(spec)
    assert cls.output_variables == ["u"]
    # PINA exposes input_variables on instances (derived from the
    # CartesianDomain axes), not on the class — instantiate to check.
    instance = cls()
    assert set(instance.input_variables or []) == {"x", "y", "z"}


def test_compose_rejects_unknown_equation_reference():
    spec = ProblemSpec(
        output_variables=["u"],
        domain_bounds={"x": [0.0, 1.0]},
        subdomains=[SubdomainSpec(name="D", bounds={"x": [0.0, 1.0]})],
        equations=[],
        conditions=[
            ConditionSpec(subdomain="D", kind="equation", equation_name="nope"),
        ],
    )
    with pytest.raises(ValueError, match="nope"):
        compose_problem(spec)


def test_compose_rejects_fixed_value_without_value():
    spec = ProblemSpec(
        output_variables=["u"],
        domain_bounds={"x": [0.0, 1.0]},
        subdomains=[SubdomainSpec(name="D", bounds={"x": [0.0, 1.0]})],
        conditions=[ConditionSpec(subdomain="D", kind="fixed_value")],
    )
    with pytest.raises(ValueError, match="fixed_value"):
        compose_problem(spec)


def test_build_equation_returns_pina_equation():
    """Happy-path build — the returned object is a pina.Equation wrapper."""
    from pina.equation import Equation

    spec = EquationSpec(
        name="linear",
        form="u - (2.0*x + 1.0)",
        outputs=["u"],
        derivatives=[],
        parameters={},
    )
    eq = build_equation(spec)
    assert isinstance(eq, Equation)


def test_build_equation_inline_in_condition():
    """Inline EquationSpec on a ConditionSpec resolves at compose time."""
    spec = ProblemSpec(
        output_variables=["u"],
        domain_bounds={"x": [0.0, 1.0]},
        subdomains=[SubdomainSpec(name="D", bounds={"x": [0.0, 1.0]})],
        conditions=[
            ConditionSpec(
                subdomain="D",
                kind="equation",
                equation_inline=EquationSpec(
                    name="inline",
                    form="u - 3.0",
                    outputs=["u"],
                    derivatives=[],
                    parameters={},
                ),
            ),
        ],
    )
    cls = compose_problem(spec)
    assert "D" in cls.conditions


def test_composed_burgers_trains_end_to_end():
    """Full integration: compose → pick model → pick solver → train 2 epochs.

    Verifies the composer's torch callable is autograd-compatible and that
    PINA's Trainer can actually optimise against it without any hardcoded
    PDE factory on the path.
    """
    from marimo_flow.core import ModelManager, SolverManager, train_solver

    cls = compose_problem(_burgers_spec("burgers_train_smoke"))
    problem = cls()
    model = ModelManager.create("feedforward", problem=problem, layers=[8, 8])
    solver = SolverManager.create(
        "pinn", problem=problem, model=model, learning_rate=1e-3
    )
    trainer = train_solver(
        solver, max_epochs=2, accelerator="cpu", n_points=64, sample_mode="random"
    )
    metrics = trainer.callback_metrics
    # Each condition gets its own loss term — they all evaluated.
    for key in ("left_loss", "right_loss", "t0_loss", "D_loss"):
        assert key in metrics, f"missing {key} in {dict(metrics)}"
        assert torch.isfinite(metrics[key]), f"{key} is not finite: {metrics[key]}"


def test_compose_emits_meaningful_class_name():
    cls = compose_problem(_burgers_spec(name="my_special_burgers"))
    assert cls.__name__ == "my_special_burgers"

    anon = compose_problem(
        ProblemSpec(
            output_variables=["u"],
            domain_bounds={"x": [0.0, 1.0]},
            subdomains=[SubdomainSpec(name="D", bounds={"x": [0.0, 1.0]})],
            conditions=[
                ConditionSpec(subdomain="D", kind="fixed_value", value=0.0)
            ],
        )
    )
    assert anon.__name__ == "ComposedProblem"
