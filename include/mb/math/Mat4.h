#pragma once
#include "Vec3.h"
#include "Mat3.h"
#include <array>

namespace mb {

class Quaternion; // forward declaration

/**
 * 4×4 Matrix (column-major storage) - Homogeneous transformation
 */
class Mat4 {
public:
    std::array<double, 16> data;

    Mat4() : data{} {}
    Mat4(const std::array<double, 16>& d) : data(d) {}

    double get(int row, int col) const { return data[col * 4 + row]; }
    void set(int row, int col, double v) { data[col * 4 + row] = v; }

    static Mat4 identity();
    static Mat4 zero();
    static Mat4 fromRotationTranslation(const Mat3& R, const Vec3& t);
    static Mat4 fromQuaternionTranslation(const Quaternion& q, const Vec3& t);

    Vec3 transformPoint(const Vec3& p) const;
    Vec3 transformDirection(const Vec3& d) const;
    Mat3 getRotation() const;
    Vec3 getTranslation() const;
    Mat4 multiply(const Mat4& other) const;
    Mat4 inverseRigid() const;
    Mat4 inverse() const; // General 4x4 inverse (Gauss-Jordan)

    Mat4 operator*(const Mat4& o) const { return multiply(o); }
};

} // namespace mb
