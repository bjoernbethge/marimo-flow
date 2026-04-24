"""Smoke tests for marimo_flow.core.viz3d — plotly-only 3D helpers."""

from __future__ import annotations

import numpy as np

from marimo_flow.agents.schemas import (
    ConditionSpec,
    DerivativeSpec,
    EquationSpec,
    MeshSpec,
    ProblemSpec,
    SubdomainSpec,
)
from marimo_flow.agents.services.composer import compose_problem
from marimo_flow.core.viz3d import (
    domain_figure,
    scatter_samples,
    volume_figure,
)


def _write_triangle_mesh(path):
    import meshio

    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    cells = [("triangle", np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64))]
    meshio.write_points_cells(str(path), points, cells)


def test_domain_figure_for_cartesian_3d_problem():
    spec = ProblemSpec(
        name="poisson_3d",
        output_variables=["u"],
        domain_bounds={"x": [0.0, 1.0], "y": [0.0, 1.0], "z": [0.0, 1.0]},
        subdomains=[
            SubdomainSpec(
                name="D",
                bounds={"x": [0.0, 1.0], "y": [0.0, 1.0], "z": [0.0, 1.0]},
            )
        ],
        equations=[
            EquationSpec(
                name="poisson",
                form="u_xx + u_yy + u_zz",
                outputs=["u"],
                derivatives=[
                    DerivativeSpec(name="u_xx", field="u", wrt=["x", "x"]),
                    DerivativeSpec(name="u_yy", field="u", wrt=["y", "y"]),
                    DerivativeSpec(name="u_zz", field="u", wrt=["z", "z"]),
                ],
            )
        ],
        conditions=[
            ConditionSpec(subdomain="D", kind="equation", equation_name="poisson"),
        ],
    )
    problem = compose_problem(spec)()
    fig = domain_figure(problem)
    assert fig.data
    assert fig.data[0].type in {"scatter3d", "mesh3d"}


def test_domain_figure_for_mesh_problem(tmp_path):
    path = tmp_path / "unit_square.vtu"
    _write_triangle_mesh(path)
    spec = ProblemSpec(
        name="poisson_mesh",
        output_variables=["u"],
        domain_bounds={"x": [0.0, 1.0], "y": [0.0, 1.0]},
        mesh=MeshSpec(
            path=str(path), axes=["x", "y", "z"], primary_cell_kind="triangle"
        ),
        subdomains=[SubdomainSpec(name="D", mesh_ref="all")],
        equations=[
            EquationSpec(
                name="poisson",
                form="u_xx + u_yy",
                outputs=["u"],
                derivatives=[
                    DerivativeSpec(name="u_xx", field="u", wrt=["x", "x"]),
                    DerivativeSpec(name="u_yy", field="u", wrt=["y", "y"]),
                ],
            )
        ],
        conditions=[
            ConditionSpec(subdomain="D", kind="equation", equation_name="poisson"),
        ],
    )
    # mesh_ref="all" is not declared — the composer should error, so we
    # register the tag and retry.
    spec.mesh.cell_tags = {"all": [0, 1]}
    problem = compose_problem(spec)()
    fig = domain_figure(problem)
    assert fig.data
    assert fig.data[0].type == "mesh3d"


def test_scatter_samples_pads_to_3d():
    pts = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    fig = scatter_samples(pts, axes=["x", "y"])
    assert fig.data[0].type == "scatter3d"
    assert len(fig.data[0].z) == 3


def test_volume_figure_smoke():
    xs = np.linspace(0, 1, 4)
    grid = np.array([(x, y, z) for x in xs for y in xs for z in xs])
    vals = (grid**2).sum(axis=1)
    fig = volume_figure(grid, vals, axes=["x", "y", "z"])
    assert fig.data[0].type == "volume"
