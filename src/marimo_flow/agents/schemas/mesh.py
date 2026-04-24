"""MeshSpec — declarative reference to an unstructured mesh.

Agents add a ``MeshSpec`` to a ``ProblemSpec`` when the domain cannot
be described by tensor-product intervals (boxes, cylinders-as-ellipsoids).
The composer resolves the mesh via ``meshio.read`` and wraps it in a
``MeshDomain`` adapter that implements PINA's ``DomainInterface`` by
barycentric sampling on the declared cell-block.

Physical tags (``point_tags`` / ``cell_tags``) let a ``SubdomainSpec``
reference a named region (``mesh_ref="inlet"``) instead of an axis box.
The composer looks up the tag to sample only the matching cells.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MeshFormat = Literal["stl", "obj", "vtk", "vtu", "gmsh", "msh", "ply", "xdmf", "med"]

CellKind = Literal[
    "triangle", "quad", "tetra", "hexahedron", "wedge", "pyramid", "line"
]


class MeshSpec(BaseModel):
    """Reference to a mesh file that the composer turns into a domain.

    ``axes`` lists the coordinate axes stored in the mesh (a surface
    mesh in 3D still has ``axes=["x","y","z"]`` — the points are 3D,
    only the cells are 2D). The composer trusts these names when
    emitting ``LabelTensor`` samples.

    ``primary_cell_kind`` picks which cell block meshio returns first
    when the file contains several (e.g. a GMSH with lines + triangles
    + tetrahedra). ``None`` means "largest block wins".

    Physical tags map human names to cell-block indices (``cell_tags``)
    or point indices (``point_tags``). Populate them once on ingest;
    the composer consults them when a ``SubdomainSpec`` sets ``mesh_ref``.
    """

    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="filesystem path to the mesh file")
    format: MeshFormat | None = Field(
        default=None,
        description="override meshio's auto-detect (by file extension)",
    )
    axes: list[str] = Field(
        default_factory=lambda: ["x", "y", "z"],
        description="axis labels; length must match the point coord dim",
    )
    primary_cell_kind: CellKind | None = Field(
        default=None,
        description="cell block to sample from; None = pick the largest",
    )
    units: str | None = Field(
        default=None,
        description="human hint for the mesh units (mm / m / …); no rescaling",
    )
    cell_tags: dict[str, list[int]] = Field(
        default_factory=dict,
        description="human-name → list of cell-block row indices",
    )
    point_tags: dict[str, list[int]] = Field(
        default_factory=dict,
        description="human-name → list of point indices (for data Conditions)",
    )
