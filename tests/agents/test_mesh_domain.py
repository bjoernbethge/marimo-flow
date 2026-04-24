"""Tests for MeshDomain + compose_problem integration via MeshSpec."""

from __future__ import annotations

import numpy as np
import pytest

from marimo_flow.agents.schemas import (
    ConditionSpec,
    DerivativeSpec,
    EquationSpec,
    MeshSpec,
    ProblemSpec,
    SubdomainSpec,
)
from marimo_flow.agents.services.composer import compose_problem
from marimo_flow.agents.services.mesh_domain import MeshDomain, load_mesh_domain


def _write_simple_triangle_mesh(path):
    """Write a minimal VTK-compatible XDMF/VTU mesh with 4 triangles."""
    import meshio

    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],
        ],
        dtype=np.float64,
    )
    cells = [
        (
            "triangle",
            np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]], dtype=np.int64),
        ),
    ]
    meshio.write_points_cells(str(path), points, cells)


def test_mesh_domain_samples_inside_bounding_box(tmp_path):
    path = tmp_path / "unit_square.vtu"
    _write_simple_triangle_mesh(path)

    spec = MeshSpec(path=str(path), axes=["x", "y", "z"], primary_cell_kind="triangle")
    dom = load_mesh_domain(spec)
    samples = dom.sample(256)
    pts = samples.tensor.detach().cpu().numpy()
    assert pts.shape == (256, 3)
    assert pts[:, 0].min() >= -1e-6 and pts[:, 0].max() <= 1.0 + 1e-6
    assert pts[:, 1].min() >= -1e-6 and pts[:, 1].max() <= 1.0 + 1e-6
    assert np.allclose(pts[:, 2], 0.0, atol=1e-6)


def test_mesh_domain_exposes_variables_and_range(tmp_path):
    path = tmp_path / "unit_square.vtu"
    _write_simple_triangle_mesh(path)
    spec = MeshSpec(path=str(path), axes=["x", "y", "z"])
    dom = load_mesh_domain(spec)
    assert dom.variables == ["x", "y", "z"]
    rng = dom.range
    assert rng["x"] == (0.0, 1.0)
    assert rng["y"] == (0.0, 1.0)


def test_mesh_domain_tag_subset_samples_only_tagged_cells(tmp_path):
    path = tmp_path / "unit_square.vtu"
    _write_simple_triangle_mesh(path)

    spec = MeshSpec(
        path=str(path),
        axes=["x", "y", "z"],
        primary_cell_kind="triangle",
        cell_tags={"bottom": [0]},
    )
    dom = load_mesh_domain(spec, mesh_ref="bottom")
    pts = dom.sample(200).tensor.detach().cpu().numpy()
    # Cell 0 = (0,0)-(1,0)-(0.5,0.5) — bottom wedge, all samples have y ≤ 0.5.
    assert pts[:, 1].max() <= 0.5 + 1e-6


def test_mesh_domain_only_supports_random_sampling(tmp_path):
    path = tmp_path / "unit_square.vtu"
    _write_simple_triangle_mesh(path)
    dom = load_mesh_domain(MeshSpec(path=str(path), axes=["x", "y", "z"]))
    with pytest.raises(ValueError, match="random"):
        dom.sample(10, mode="grid")


def test_compose_problem_with_mesh_subdomain_composes_successfully(tmp_path):
    """Full flow: spec attaches a mesh → subdomain uses mesh_ref → composer wires MeshDomain."""
    path = tmp_path / "unit_square.vtu"
    _write_simple_triangle_mesh(path)

    spec = ProblemSpec(
        name="poisson_mesh",
        output_variables=["u"],
        domain_bounds={"x": [0.0, 1.0], "y": [0.0, 1.0]},
        mesh=MeshSpec(
            path=str(path),
            axes=["x", "y", "z"],
            primary_cell_kind="triangle",
            cell_tags={"interior": [0, 1, 2, 3]},
        ),
        subdomains=[
            SubdomainSpec(name="D", mesh_ref="interior"),
        ],
        equations=[
            EquationSpec(
                name="poisson",
                form="u_xx + u_yy",
                outputs=["u"],
                derivatives=[
                    DerivativeSpec(name="u_xx", field="u", wrt=["x", "x"]),
                    DerivativeSpec(name="u_yy", field="u", wrt=["y", "y"]),
                ],
            ),
        ],
        conditions=[
            ConditionSpec(subdomain="D", kind="equation", equation_name="poisson"),
        ],
    )
    cls = compose_problem(spec)
    assert "D" in cls.domains
    assert isinstance(cls.domains["D"], MeshDomain)


def test_compose_rejects_mesh_ref_without_mesh():
    spec = ProblemSpec(
        output_variables=["u"],
        domain_bounds={"x": [0.0, 1.0]},
        subdomains=[SubdomainSpec(name="D", mesh_ref="somewhere")],
        conditions=[ConditionSpec(subdomain="D", kind="fixed_value", value=0.0)],
    )
    with pytest.raises(ValueError, match="mesh_ref"):
        compose_problem(spec)
