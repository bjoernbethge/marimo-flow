"""3D visualisation helpers for PINA problems — plotly-only.

Used by the provenance dashboard to render:

* the spatial domain of a composed problem (CartesianDomain → wireframe
  box; MeshDomain → triangle Mesh3d surface);
* collocation samples that a trained solver draws (Scatter3d);
* predicted fields on a volumetric grid (Volume or Isosurface).

No pyvista / VTK — plotly ships in the project's existing deps and
renders inline inside marimo notebooks.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def domain_figure(problem: Any) -> Any:
    """Return a plotly figure showing the problem's spatial domain."""

    try:
        spatial = problem.spatial_domain
    except AttributeError as exc:
        raise ValueError(
            "problem has no spatial_domain — cannot visualise in 3D"
        ) from exc

    if hasattr(spatial, "vert_matrix"):  # SimplexDomain
        return _simplex_figure(spatial)
    if _is_mesh_domain(spatial):
        return _mesh_figure(spatial)
    return _box_figure(spatial)


def scatter_samples(points: np.ndarray, axes: list[str]) -> Any:
    """Render sampled collocation points as a Scatter3d."""
    import plotly.graph_objects as go

    pts = np.asarray(points)
    if pts.shape[1] < 3:
        pad = 3 - pts.shape[1]
        pts = np.concatenate([pts, np.zeros((len(pts), pad))], axis=1)
    axes = axes[:3] + ["z"] * max(0, 3 - len(axes))
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                marker={"size": 2, "opacity": 0.6},
            )
        ]
    )
    fig.update_layout(
        scene={"xaxis_title": axes[0], "yaxis_title": axes[1], "zaxis_title": axes[2]}
    )
    return fig


def volume_figure(
    grid_points: np.ndarray,
    values: np.ndarray,
    axes: list[str],
    *,
    isomin: float | None = None,
    isomax: float | None = None,
    surface_count: int = 12,
) -> Any:
    """Render a scalar volume as a translucent plotly Volume trace.

    ``grid_points`` is ``(N, 3)`` and ``values`` is ``(N,)`` over a
    regular 3D grid. For irregular samples, caller should interpolate
    first.
    """
    import plotly.graph_objects as go

    pts = np.asarray(grid_points)
    vals = np.asarray(values).flatten()
    if isomin is None:
        isomin = float(np.nanmin(vals))
    if isomax is None:
        isomax = float(np.nanmax(vals))
    fig = go.Figure(
        data=go.Volume(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            value=vals,
            isomin=isomin,
            isomax=isomax,
            opacity=0.1,
            surface_count=surface_count,
        )
    )
    fig.update_layout(
        scene={
            "xaxis_title": axes[0],
            "yaxis_title": axes[1],
            "zaxis_title": axes[2],
        }
    )
    return fig


# --- internals -----------------------------------------------------


def _is_mesh_domain(spatial: Any) -> bool:
    from marimo_flow.agents.services.mesh_domain import MeshDomain

    return isinstance(spatial, MeshDomain)


def _mesh_figure(mesh: Any) -> Any:
    import plotly.graph_objects as go

    pts = mesh.points
    cells = mesh.cells
    if cells.shape[1] == 3:
        i, j, k = cells[:, 0], cells[:, 1], cells[:, 2]
        trace = go.Mesh3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            i=i,
            j=j,
            k=k,
            opacity=0.5,
            color="lightblue",
        )
    else:
        # Tetra / hex: fall back to vertex scatter.
        trace = go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode="markers",
            marker={"size": 2},
        )
    fig = go.Figure(data=[trace])
    return fig


def _box_figure(cartesian: Any) -> Any:
    """Draw the wireframe of a CartesianDomain box (3D or padded to 3D)."""
    import plotly.graph_objects as go

    ranges = cartesian.range
    axes = list(ranges.keys())[:3]
    while len(axes) < 3:
        axes.append(f"pad{len(axes)}")
        ranges[axes[-1]] = (0.0, 0.0)
    lo = [ranges[a][0] for a in axes]
    hi = [ranges[a][1] for a in axes]
    # 8 corners of the axis-aligned box.
    verts = np.array(
        [
            [lo[0], lo[1], lo[2]],
            [hi[0], lo[1], lo[2]],
            [hi[0], hi[1], lo[2]],
            [lo[0], hi[1], lo[2]],
            [lo[0], lo[1], hi[2]],
            [hi[0], lo[1], hi[2]],
            [hi[0], hi[1], hi[2]],
            [lo[0], hi[1], hi[2]],
        ]
    )
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    xs, ys, zs = [], [], []
    for a, b in edges:
        xs.extend([verts[a, 0], verts[b, 0], None])
        ys.extend([verts[a, 1], verts[b, 1], None])
        zs.extend([verts[a, 2], verts[b, 2], None])
    fig = go.Figure(
        data=[go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line={"width": 4})]
    )
    fig.update_layout(
        scene={
            "xaxis_title": axes[0],
            "yaxis_title": axes[1],
            "zaxis_title": axes[2],
        }
    )
    return fig


def _simplex_figure(simplex: Any) -> Any:
    import plotly.graph_objects as go

    vertex_tensors = simplex.vert_matrix
    verts = np.vstack([v.tensor.detach().cpu().numpy() for v in vertex_tensors])
    if verts.shape[1] < 3:
        verts = np.concatenate(
            [verts, np.zeros((verts.shape[0], 3 - verts.shape[1]))], axis=1
        )
    # Close the simplex by drawing every pair of vertices.
    xs, ys, zs = [], [], []
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            xs.extend([verts[i, 0], verts[j, 0], None])
            ys.extend([verts[i, 1], verts[j, 1], None])
            zs.extend([verts[i, 2], verts[j, 2], None])
    return go.Figure(data=[go.Scatter3d(x=xs, y=ys, z=zs, mode="lines")])
