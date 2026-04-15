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
)

add_library(multibody STATIC ${LIB_SOURCES})
target_include_directories(multibody PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# ---- Example executable ----
add_executable(pendulum_example examples/pendulum.cpp)
target_link_libraries(pendulum_example PRIVATE multibody)

# ---- Qt visualization (optional) ----
find_package(Qt5 COMPONENTS Widgets QUIET)
if(Qt5_FOUND)
    set(CMAKE_AUTOMOC ON)
    add_executable(pendulum_qt examples/pendulum_qt.cpp)
    target_link_libraries(pendulum_qt PRIVATE multibody Qt5::Widgets)
    message(STATUS "Qt5 found – building pendulum_qt")
else()
    message(STATUS "Qt5 not found – skipping pendulum_qt")
endif()

# ---- Optional: install ----
install(TARGETS multibody
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
install(DIRECTORY include/mb DESTINATION include)
