cmake_minimum_required(VERSION 3.16)
project(MultibodyCpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Library sources ----
set(LIB_SOURCES
    # Math
    src/math/Vec3.cpp
    src/math/Mat3.cpp
    src/math/Mat4.cpp
    src/math/Quaternion.cpp
    src/math/MatrixN.cpp
    src/math/SparseMatrix.cpp

    # Core
    src/core/Body.cpp
    src/core/RigidBody.cpp
    src/core/ReferenceFrame.cpp
    src/core/State.cpp

    # Constraints
    src/constraints/Constraint.cpp
    src/constraints/SphericalJoint.cpp
    src/constraints/RevoluteJoint.cpp
    src/constraints/PrismaticJoint.cpp
    src/constraints/FixedJoint.cpp
    src/constraints/DistanceConstraint.cpp
    src/constraints/DrivingConstraint.cpp

    # Forces
    src/forces/Gravity.cpp
    src/forces/AppliedForce.cpp
    src/forces/SpringDamper.cpp

    # Solvers
    src/solvers/ConstraintSolver.cpp
    src/solvers/DirectSolver.cpp
    src/solvers/NewtonRaphson.cpp

    # Integrators
    src/integrators/TimeIntegrator.cpp
    src/integrators/RungeKutta.cpp
    src/integrators/BDF.cpp

    # Contact
    src/contact/CollisionDetector.cpp
    src/contact/ContactForceModel.cpp
    src/contact/ContactManager.cpp

    # System
    src/system/MultibodySystem.cpp

    # FEM / Flexible Bodies
    src/fem/ElasticMaterial.cpp
    src/fem/ANCFTetrahedralElement.cpp
    src/fem/FlexibleBody.cpp
    src/fem/GmshReader.cpp
    src/fem/FlexibleContactDetector.cpp
    src/fem/FlexibleContactForce.cpp
    src/fem/FlexibleContactManager.cpp
    src/fem/FlexibleIntegrators.cpp
)

add_library(multibody STATIC ${LIB_SOURCES})
target_include_directories(multibody PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# ---- Example executable ----
add_executable(pendulum_example examples/pendulum.cpp)
target_link_libraries(pendulum_example PRIVATE multibody)

add_executable(flex_energy_test examples/flex_energy_test.cpp)
target_link_libraries(flex_energy_test PRIVATE multibody)

# ---- Qt visualization (optional) ----
find_package(Qt5 COMPONENTS Widgets QUIET)
if(Qt5_FOUND)
    set(CMAKE_AUTOMOC ON)
    add_executable(pendulum_qt examples/pendulum_qt.cpp)
    target_link_libraries(pendulum_qt PRIVATE multibody Qt5::Widgets)

    add_executable(chain20_qt examples/chain20_qt.cpp)
    target_link_libraries(chain20_qt PRIVATE multibody Qt5::Widgets)

    add_executable(cantilever_qt examples/cantilever_qt.cpp)
    target_link_libraries(cantilever_qt PRIVATE multibody Qt5::Widgets)

    add_executable(spinning_beam_qt examples/spinning_beam_qt.cpp)
    target_link_libraries(spinning_beam_qt PRIVATE multibody Qt5::Widgets)

    add_executable(tet_collision_qt examples/tet_collision_qt.cpp)
    target_link_libraries(tet_collision_qt PRIVATE multibody Qt5::Widgets)

    message(STATUS "Qt5 found – building pendulum_qt, chain20_qt, cantilever_qt, spinning_beam_qt, tet_collision_qt")
else()
    message(STATUS "Qt5 not found – skipping pendulum_qt")
endif()

# ---- Optional: install ----
install(TARGETS multibody
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
install(DIRECTORY include/mb DESTINATION include)
