---
name: pina-geometry
description: Use an unstructured mesh (STL/OBJ/VTK/GMSH) as the spatial domain for a PINA problem — attach a MeshSpec and reference tagged cell regions via SubdomainSpec.mesh_ref.
triggers:
  - mesh pde
  - mesh geometry
  - stl
  - obj
  - vtk
  - gmsh
  - xdmf
  - cad geometry
  - complex geometry
  - non-rectangular domain
  - airfoil
  - engine block
  - heart chamber
---

# PINA on Unstructured Meshes

Use this skill when the problem's spatial domain **can not** be described
by axis-aligned boxes (CartesianDomain) or ellipsoids. Typical cases:

- Car body, turbine blade, airfoil (external aerodynamics).
- Heart chamber, lung airway, vasculature (biomedical).
- Engine block, heat exchanger (thermal management).
- Any geometry that was meshed in Gmsh / OpenFOAM / Ansys.

## Requirements

- **Inputs**: a mesh file in one of ``stl`` / ``obj`` / ``vtk`` / ``vtu``
  / ``gmsh`` / ``msh`` / ``ply`` / ``xdmf`` / ``med``. PINA receives the
  raw points + cells through meshio.
- **Cells**: only ``triangle`` (2D), ``tetra`` (3D), plus ``quad``/``hex``
  (decomposed internally) are supported for sampling. ``line``-only
  meshes work but are rarely useful.
- **CAD (STEP/IGES)**: not supported out of the box — convert to STL
  first via Gmsh or FreeCAD. A future CAD bridge via ``pythonocc-core``
  is on the roadmap (Phase C optional).

## Recipe

1. **Attach a MeshSpec to the ProblemSpec**:

   ```python
   from marimo_flow.agents.schemas import MeshSpec

   mesh = MeshSpec(
       path="data/wing.stl",
       axes=["x", "y", "z"],
       primary_cell_kind="triangle",
       cell_tags={
           "surface": list(range(0, 12000)),       # outer skin
           "trailing_edge": [12000, 12001, ..., 12099],
       },
   )
   ```

   The ``cell_tags`` dict is a hand-coded mapping from human names to
   cell-row indices. For Gmsh files this comes from physical-group
   info — the Data agent can enumerate those (``meshio.read(...).cell_data``).

2. **Reference tags from subdomains**:

   ```python
   SubdomainSpec(name="D",         mesh_ref="interior"),   # bulk PDE
   SubdomainSpec(name="inlet",     mesh_ref="inlet"),      # Dirichlet BC
   SubdomainSpec(name="outlet",    mesh_ref="outlet"),     # Neumann / fixed_value
   ```

3. **Compose** — the composer builds a ``MeshDomain`` per subdomain and
   wires them as ``problem.domains``. Inference and training use
   barycentric random sampling on the tagged cells.

## End-to-end example — Poisson on a tagged VTU

```python
from marimo_flow.agents.schemas import (
    ConditionSpec, DerivativeSpec, EquationSpec,
    MeshSpec, ProblemSpec, SubdomainSpec,
)
from marimo_flow.agents.services.composer import compose_problem

spec = ProblemSpec(
    name="poisson_on_mesh",
    output_variables=["u"],
    domain_bounds={"x": [0.0, 1.0], "y": [0.0, 1.0]},  # rough bbox hint
    mesh=MeshSpec(
        path="data/unit_square.vtu",
        axes=["x", "y", "z"],
        primary_cell_kind="triangle",
        cell_tags={"interior": list(range(0, 4))},
    ),
    subdomains=[SubdomainSpec(name="D", mesh_ref="interior")],
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
```

## Gotchas

- **Cell-kind uniformity**: a single ``MeshDomain`` samples from one
  cell kind at a time. For meshes that mix ``triangle`` + ``line``
  (surface + edges), ``primary_cell_kind`` decides.
- **Z-axis of 2D meshes**: meshio always stores 3D points even for
  triangle meshes. Declare ``axes=["x","y","z"]`` and let the model
  ignore z (output is still a function of x,y).
- **Non-closed surfaces** (STL of an open sheet): sampling still works
  but the PDE residual must only reference tangential derivatives.
- **Very large meshes** (>1 M cells): the point array is held in
  memory. For bigger meshes consider down-sampling via meshio's
  ``prune`` + ``reindex`` before registering.
- **BCs**: PINA's standard ``FixedValue`` works on any PINA domain,
  including ``MeshDomain``. For Neumann, write an inline equation that
  computes the normal derivative — the Problem agent must add the
  outward normal via a ``DerivativeSpec`` per axis.

## Escalation

If the mesh has holes / flipped normals / duplicate points (common in
CAD-exported STLs), the mesh must be repaired **before** ingest. Use
``meshio.read(path).prune_z_0()`` for planar 2D or invoke Gmsh/FreeCAD
manually. Don't try to patch this inside the composer.
