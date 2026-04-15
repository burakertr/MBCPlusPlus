#pragma once
#include "Vec3.h"
#include "Mat3.h"
#include <cmath>

namespace mb {

class MatrixN; // forward declaration

/**
 * Quaternion (Hamilton convention: w, x, y, z)
 * q = w + xi + yj + zk
 */
class Quaternion {
public:
    double w, x, y, z;

    Quaternion() : w(1), x(0), y(0), z(0) {}
    Quaternion(double w, double x, double y, double z) : w(w), x(x), y(y), z(z) {}

    // Factory methods
    static Quaternion identity() { return {1, 0, 0, 0}; }
    static Quaternion fromAxisAngle(const Vec3& axis, double angle);
    static Quaternion fromEulerZYX(double z, double y, double x);
    static Quaternion fromRotationMatrix(const Mat3& R);
    static Quaternion fromVectors(const Vec3& from, const Vec3& to);

    // Operations
    Quaternion multiply(const Quaternion& q) const;
    Vec3 rotate(const Vec3& v) const;
    Vec3 inverseRotate(const Vec3& v) const;
    Mat3 toRotationMatrix() const;
    Quaternion conjugate() const { return {w, -x, -y, -z}; }
    Quaternion inverse() const;
    Quaternion normalize() const;
    double norm() const { return std::sqrt(w*w + x*x + y*y + z*z); }
    double dot(const Quaternion& q) const { return w*q.w + x*q.x + y*q.y + z*q.z; }

    // Quaternion derivative from angular velocity (world frame)
    // q̇ = 0.5 * [0, ω_world] ⊗ q
    Quaternion derivative(const Vec3& omega) const;

    // G matrix: maps angular velocity to quaternion derivative
    // q̇ = 0.5 * G(q) * ω
    MatrixN getGMatrix() const;

    // L matrix (left multiplication matrix)
    MatrixN getLMatrix() const;

    // R matrix (right multiplication matrix) 
    MatrixN getRMatrix() const;

    static Quaternion slerp(const Quaternion& a, const Quaternion& b, double t);

    Quaternion operator*(const Quaternion& q) const { return multiply(q); }

    Quaternion clone() const { return *this; }
};

} // namespace mb
