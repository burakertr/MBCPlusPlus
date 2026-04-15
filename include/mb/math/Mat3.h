#pragma once
#include "Vec3.h"
#include <array>

namespace mb {

/**
 * 3×3 Matrix (column-major storage)
 */
class Mat3 {
public:
    // Column-major: data[col * 3 + row]
    std::array<double, 9> data;

    Mat3() : data{} {}
    Mat3(const std::array<double, 9>& d) : data(d) {}

    // Element access (row, col), column-major
    double get(int row, int col) const { return data[col * 3 + row]; }
    void set(int row, int col, double v) { data[col * 3 + row] = v; }

    // Factory methods
    static Mat3 zero() { Mat3 m; m.data.fill(0); return m; }
    static Mat3 identity() {
        Mat3 m;
        m.data.fill(0);
        m.data[0] = m.data[4] = m.data[8] = 1.0;
        return m;
    }
    static Mat3 diagonal(double a, double b, double c) {
        Mat3 m;
        m.data.fill(0);
        m.data[0] = a; m.data[4] = b; m.data[8] = c;
        return m;
    }
    static Mat3 fromSkewSymmetric(const Vec3& v) {
        Mat3 m;
        m.data.fill(0);
        m.set(0, 1, -v.z); m.set(0, 2,  v.y);
        m.set(1, 0,  v.z); m.set(1, 2, -v.x);
        m.set(2, 0, -v.y); m.set(2, 1,  v.x);
        return m;
    }

    static Mat3 fromRotationX(double angle);
    static Mat3 fromRotationY(double angle);
    static Mat3 fromRotationZ(double angle);

    // Operations
    Mat3 multiply(const Mat3& other) const;
    Vec3 multiplyVec3(const Vec3& v) const {
        return {
            get(0,0)*v.x + get(0,1)*v.y + get(0,2)*v.z,
            get(1,0)*v.x + get(1,1)*v.y + get(1,2)*v.z,
            get(2,0)*v.x + get(2,1)*v.y + get(2,2)*v.z
        };
    }
    // Alias for multiplyVec3
    Vec3 transform(const Vec3& v) const { return multiplyVec3(v); }

    Mat3 transpose() const;
    Mat3 inverse() const;
    double determinant() const;
    Mat3 add(const Mat3& other) const;
    Mat3 sub(const Mat3& other) const;
    Mat3 scale(double s) const;

    // Operators
    Mat3 operator*(const Mat3& o) const { return multiply(o); }
    Vec3 operator*(const Vec3& v) const { return multiplyVec3(v); }
    Mat3 operator+(const Mat3& o) const { return add(o); }
    Mat3 operator-(const Mat3& o) const { return sub(o); }
    Mat3 operator*(double s) const { return scale(s); }

    Vec3 getColumn(int col) const {
        return {data[col*3], data[col*3+1], data[col*3+2]};
    }
    void setColumn(int col, const Vec3& v) {
        data[col*3] = v.x;
        data[col*3+1] = v.y;
        data[col*3+2] = v.z;
    }
};

} // namespace mb
