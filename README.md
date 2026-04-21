# MBC++ — Multibody & Flexible-Body Dynamics in C++

A modern C++17 library for **rigid multibody dynamics** and **flexible-body finite-element analysis** using the Absolute Nodal Coordinate Formulation (ANCF). Includes real-time Qt5 visualization, contact/collision handling, and a rich set of time integrators.

<p align="center">
  <img src="output.gif" alt="MBC++ Simulation Example" width="700"/>
</p>

<p align="center">
  <img src="output2.gif" alt="MBC++ Cantilever Beam Simulation" width="700"/>
</p>

---

## Features

- **Rigid-body dynamics** — constrained multibody systems with joints, forces, and contact
- **Flexible-body dynamics** — ANCF tetrahedral elements with St. Venant–Kirchhoff and Neo-Hookean materials
- **Contact & collision** — broad-phase AABB + narrow-phase detection for both rigid and flexible bodies; Hertz + Flores + Coulomb force models
- **Multiple integrators** — RK4, Dormand–Prince 4(5), BDF-1/2, Generalized-α, Semi-Implicit Euler, Velocity Verlet
- **Constraint solvers** — Direct solver, Newton–Raphson with GGL stabilization
- **Qt5 visualization** — interactive 3D viewers with orbit/pan/zoom and keyboard controls
- **MP4 recording** — real-time 60 fps video capture via ffmpeg pipe
- **Gmsh mesh import** — read `.msh` files directly into ANCF models

---

## Project Structure

```
MBC++/
├── CMakeLists.txt
├── include/mb/           # Public headers
│   ├── core/             # Body, RigidBody, ReferenceFrame, State
│   ├── math/             # Vec3, Mat3, Mat4, Quaternion, MatrixN, SparseMatrix, VectorN
│   ├── constraints/      # Constraint base, joints (Spherical, Revolute, Prismatic, Fixed, Distance, Driving)
│   ├── forces/           # Force base, Gravity, AppliedForce, AppliedTorque, SpringDamper
│   ├── integrators/      # TimeIntegrator base, RK4, RKF45, DOPRI45, BDF1/2, Gen-α, etc.
│   ├── solvers/          # ConstraintSolver, DirectSolver, NewtonRaphsonSolver
│   ├── contact/          # CollisionDetector, ContactForceModel, ContactManager
│   ├── fem/              # ANCF flexible bodies, elements, materials, flexible contact pipeline
│   └── system/           # MultibodySystem (central orchestrator)
├── src/                  # Implementation files (mirrors include/ layout)
└── examples/             # Demo applications
```

---

## Modules & Classes

### Math (`mb/math/`)
| Class | Description |
|---|---|
| `Vec3` | 3D vector with arithmetic, cross/dot products, norms |
| `Mat3` | 3×3 matrix (rotations, inertia tensors) |
| `Mat4` | 4×4 homogeneous transformation matrix |
| `Quaternion` | Unit quaternion for 3D rotations |
| `MatrixN` / `VectorN` | Dense N×N matrix and N-vector for general linear algebra |
| `SparseMatrix` | Sparse matrix (triplet storage) |

### Core (`mb/core/`)
| Class | Description |
|---|---|
| `Body` | Abstract base class for all bodies (`BodyType`, `MassProperties`, `BodyState`) |
| `RigidBody` | Rigid body with mass, inertia, collision shapes (sphere, box, cylinder, cone, mesh, plane); nq=7, nv=6 |
| `ReferenceFrame` | Coordinate frame with position + orientation |
| `StateVector` | System-level state vector management |

### Constraints (`mb/constraints/`)
| Class | Description |
|---|---|
| `Constraint` | Abstract base; provides Jacobian, violation, RHS |
| `SphericalJoint` | Ball-and-socket joint (3 constraints) |
| `RevoluteJoint` | Hinge joint (5 constraints) |
| `PrismaticJoint` | Sliding joint along an axis |
| `FixedJoint` | Fully locks two bodies together |
| `DistanceConstraint` | Maintains constant distance between two points |
| `DrivingConstraint` | Prescribes motion via a time function |

### Forces (`mb/forces/`)
| Class | Description |
|---|---|
| `Force` | Abstract base; `SingleBodyForce`, `TwoBodyForce` variants |
| `Gravity` | Uniform gravitational field |
| `AppliedForce` / `AppliedTorque` | External force/torque on a single body |
| `SpringDamper` | Linear spring-damper between two bodies |

### Integrators (`mb/integrators/`)
| Class | Description |
|---|---|
| `TimeIntegrator` | Abstract base with `IntegratorConfig`, `StepResult` |
| `RungeKutta4` | Classic 4th-order Runge–Kutta |
| `RungeKuttaFehlberg45` | RKF4(5) with adaptive step control |
| `DormandPrince45` | DOPRI4(5) adaptive integrator |
| `BackwardEuler` | Implicit 1st-order BDF |
| `BDF2` | Implicit 2nd-order BDF |
| `GeneralizedAlpha` | Generalized-α method (tunable numerical damping) |
| `SemiImplicitEuler` | Symplectic Euler |
| `VelocityVerlet` | 2nd-order symplectic integrator |

### Solvers (`mb/solvers/`)
| Class | Description |
|---|---|
| `ConstraintSolver` | Abstract base with `SolverResult`, `SolverConfig` |
| `DirectSolver` | Direct factorization-based constraint solver |
| `NewtonRaphsonSolver` | Nonlinear Newton–Raphson with position-level correction |

### Contact — Rigid (`mb/contact/`)
| Class | Description |
|---|---|
| `CollisionDetector` | Broad/narrow-phase collision detection for rigid shapes |
| `ContactForceModel` | Penalty-based contact force computation |
| `ContactManager` | Orchestrates detection → force pipeline |
| `ContactMaterial` / `ContactPair` / `ActiveContact` / `ContactConfig` | Supporting data types |

### FEM / Flexible Bodies (`mb/fem/`)
| Class | Description |
|---|---|
| `FlexibleBody` | ANCF-based flexible body (inherits `Body`); manages nodes, elements, boundary conditions, mass/force assembly |
| `ANCFTetrahedralElement` | 4-node tetrahedral ANCF element (48 DOF) |
| `ANCFNode` | Node with 12 DOF (3 position + 9 deformation gradient) |
| `ElasticMaterialProps` / `LameParams` | Material property structs |
| `MaterialModel` | Abstract material interface |
| `StVenantKirchhoff` | Linear elastic hyperelastic material |
| `NeoHookean` | Large-deformation hyperelastic material |
| `GmshReader` | Imports Gmsh `.msh` meshes |
| `FlexibleBodyIntegrator` | Explicit RK4 integrator for flexible bodies |
| `FlexDOPRI45` | Adaptive Dormand–Prince 4(5) for flexible bodies |
| `ImplicitFlexIntegrator` | Implicit Newmark-β / HHT-α integrator with Newton–Raphson; unconditionally stable for stiff materials |
| `FlexibleContactDetector` | Surface extraction, AABB broad-phase, narrow-phase for deformable bodies |
| `FlexibleContactForce` | Hertz + Flores + Coulomb contact forces |
| `FlexibleContactManager` | Full contact pipeline orchestrator for flexible bodies |

### System (`mb/system/`)
| Class | Description |
|---|---|
| `MultibodySystem` | Central orchestrator — assembles M, Cq, Q, γ; integrates with constraint projection (GGL stabilization) |
| `AnalysisResult` | Energy/momentum snapshot |
| `SimStats` | Per-step simulation statistics |

---

## Build

### Requirements

- **C++17** compiler (GCC ≥ 7, Clang ≥ 5, MSVC ≥ 19.14)
- **CMake** ≥ 3.16
- **Qt5 Widgets** *(optional — for visualization examples)*

### Build Steps

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### Targets

| Target | Description |
|---|---|
| `multibody` | Static library (`libmultibody.a`) |
| `pendulum_example` | Console-only pendulum demo |
| `flex_energy_test` | Flexible-body energy conservation test |
| `pendulum_qt` | Qt5 pendulum visualization *(requires Qt5)* |
| `chain20_qt` | 20-body chain demo *(requires Qt5)* |
| `cantilever_qt` | ANCF cantilever beam *(requires Qt5)* |
| `spinning_beam_qt` | ANCF spinning beam with centrifugal stiffening *(requires Qt5)* |
| `tet_collision_qt` | Tetrahedral mesh collision demo *(requires Qt5)* |

Build a specific target:

```bash
cmake --build build --target spinning_beam_qt -j$(nproc)
```

### Install

```bash
cmake --install build --prefix /usr/local
```

Installs headers to `<prefix>/include/mb/` and the static library to `<prefix>/lib/`.

---

## Examples

| Example | Description |
|---|---|
| `pendulum.cpp` | Simple rigid-body pendulum (console output) |
| `pendulum_qt.cpp` | Interactive rigid-body pendulum with Qt5 viewer |
| `chain20_qt.cpp` | 20-link constrained chain with gravity |
| `cantilever_qt.cpp` | ANCF cantilever beam under gravity |
| `spinning_beam_qt.cpp` | Spinning ANCF beam — centrifugal stiffening demo. Controls: Space (pause), W/S (speed ω), G (toggle gravity), +/− (zoom), mouse orbit/pan |
| `tet_collision_qt.cpp` | Two ANCF tetrahedra colliding with ground contact, elastic/plastic modes, MP4 recording |
| `energy_test.cpp` | Rigid-body energy conservation verification |
| `flex_energy_test.cpp` | Flexible-body energy conservation verification |

---

## Quick Start

```cpp
#include "mb/fem/FlexibleBody.h"
#include "mb/fem/FlexibleIntegrators.h"
#include "mb/fem/ANCFTypes.h"

using namespace mb;

// Define material
ElasticMaterialProps mat{70e9, 0.3, 7800.0, MaterialType::NeoHookean};

// Build flexible body from Gmsh mesh
auto body = FlexibleBody::fromMesh(mesh, mat, "beam");
body->fixNodesOnPlane('x', 0.0);  // clamp at x = 0

// Create integrator and step
FlexDOPRI45 integrator(*body);
for (int i = 0; i < 1000; ++i) {
    FlexStepResult res = integrator.step(0.01);
}
```

---

## Contact Model Theory

The flexible-body contact pipeline uses a **Hertz–Flores** normal force with **Coulomb** friction:

### Normal Force

$$F_n = K \, \delta^{3/2} \left(1 + D \, \dot{\delta}\right)$$

where the contact stiffness $K = \frac{4}{3} E^* \sqrt{R^*}$, the combined modulus:

$$\frac{1}{E^*} = \frac{1-\nu_1^2}{E_1} + \frac{1-\nu_2^2}{E_2}$$

and the Flores dissipation coefficient:

$$D = \frac{8(1-e)}{5 \, e \, v_{\text{impact}}}$$

### Friction Force

$$\mathbf{F}_t = -\mu \, F_n \, \frac{\mathbf{v}_t}{\max(|\mathbf{v}_t|, \, v_{\text{reg}})}$$

### Implicit Integration (Newmark-β / HHT-α)

For stiff materials (steel, E ~ 200 GPa), explicit integrators require prohibitively small time steps due to the CFL condition. The `ImplicitFlexIntegrator` provides unconditional stability:

$$M \, \mathbf{a}_{n+1} = (1+\alpha) \, \mathbf{Q}_{n+1} - \alpha \, \mathbf{Q}_n$$

Solved via Newton–Raphson with central finite-difference tangent stiffness and dense LU factorization on free DOFs.

**Parameters:**
- `hhtAlpha` ∈ [−1/3, 0] — numerical damping (0 = no damping)
- `newtonTol` — Newton convergence tolerance
- `maxNewtonIter` — maximum Newton iterations per step

---

## License

This project is licensed under the GNU General Public License v3.0 (GPLv3). See the [LICENSE](LICENSE) file for details.
