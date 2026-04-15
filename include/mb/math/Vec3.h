#pragma once
#include <cmath>
#include <array>
#include <string>

namespace mb {

class Mat3; // forward declaration

/**
 * 3D Vector class with immutable and in-place operations.
 */
class Vec3 {
public:
    double x, y, z;

    Vec3() : x(0), y(0), z(0) {}
    Vec3(double x, double y, double z) : x(x), y(y), z(z) {}

    // Factory methods
    static Vec3 zero() { return {0, 0, 0}; }
    static Vec3 unitX() { return {1, 0, 0}; }
    static Vec3 unitY() { return {0, 1, 0}; }
    static Vec3 unitZ() { return {0, 0, 1}; }

    // Immutable operations (return new Vec3)
    Vec3 add(const Vec3& v) const { return {x + v.x, y + v.y, z + v.z}; }
    Vec3 sub(const Vec3& v) const { return {x - v.x, y - v.y, z - v.z}; }
    Vec3 scale(double s) const { return {x * s, y * s, z * s}; }
    Vec3 negate() const { return {-x, -y, -z}; }

    double dot(const Vec3& v) const { return x * v.x + y * v.y + z * v.z; }

    Vec3 cross(const Vec3& v) const {
        return {
            y * v.z - z * v.y,
            z * v.x - x * v.z,
            x * v.y - y * v.x
        };
    }

    double length() const { return std::sqrt(x * x + y * y + z * z); }
    double lengthSquared() const { return x * x + y * y + z * z; }

    Vec3 normalize() const {
        double len = length();
        if (len < 1e-12) return {0, 0, 0};
        double inv = 1.0 / len;
        return {x * inv, y * inv, z * inv};
    }

    Vec3 lerp(const Vec3& v, double t) const {
        return {
            x + (v.x - x) * t,
            y + (v.y - y) * t,
            z + (v.z - z) * t
        };
    }

    Vec3 clone() const { return *this; }

    // In-place operations
    Vec3& addInPlace(const Vec3& v) { x += v.x; y += v.y; z += v.z; return *this; }
    Vec3& subInPlace(const Vec3& v) { x -= v.x; y -= v.y; z -= v.z; return *this; }
    Vec3& scaleInPlace(double s) { x *= s; y *= s; z *= s; return *this; }

    // Skew-symmetric matrix [v]×
    Mat3 skewSymmetric() const;

    // Operators
    Vec3 operator+(const Vec3& v) const { return add(v); }
    Vec3 operator-(const Vec3& v) const { return sub(v); }
    Vec3 operator*(double s) const { return scale(s); }
    Vec3 operator-() const { return negate(); }
    Vec3& operator+=(const Vec3& v) { return addInPlace(v); }
    Vec3& operator-=(const Vec3& v) { return subInPlace(v); }
    Vec3& operator*=(double s) { return scaleInPlace(s); }

    double operator[](int i) const {
        switch (i) {
            case 0: return x;
            case 1: return y;
            case 2: return z;
            default: return 0;
        }
    }

    std::string toString() const;
};

inline Vec3 operator*(double s, const Vec3& v) { return v.scale(s); }

} // namespace mb
