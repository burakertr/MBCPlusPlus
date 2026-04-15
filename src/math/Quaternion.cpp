#include "mb/math/Quaternion.h"
#include "mb/math/MatrixN.h"
#include <cmath>

namespace mb {

Quaternion Quaternion::fromAxisAngle(const Vec3& axis, double angle) {
    double ha = angle / 2.0;
    double s = std::sin(ha);
    Vec3 n = axis.normalize();
    return {std::cos(ha), n.x * s, n.y * s, n.z * s};
}

Quaternion Quaternion::fromEulerZYX(double rz, double ry, double rx) {
    double cx = std::cos(rx / 2), sx = std::sin(rx / 2);
    double cy = std::cos(ry / 2), sy = std::sin(ry / 2);
    double cz = std::cos(rz / 2), sz = std::sin(rz / 2);
    return {
        cx*cy*cz + sx*sy*sz,
        sx*cy*cz - cx*sy*sz,
        cx*sy*cz + sx*cy*sz,
        cx*cy*sz - sx*sy*cz
    };
}

Quaternion Quaternion::fromRotationMatrix(const Mat3& R) {
    double trace = R.get(0,0) + R.get(1,1) + R.get(2,2);
    double w, x, y, z;

    if (trace > 0) {
        double s = 0.5 / std::sqrt(trace + 1.0);
        w = 0.25 / s;
        x = (R.get(2,1) - R.get(1,2)) * s;
        y = (R.get(0,2) - R.get(2,0)) * s;
        z = (R.get(1,0) - R.get(0,1)) * s;
    } else if (R.get(0,0) > R.get(1,1) && R.get(0,0) > R.get(2,2)) {
        double s = 2.0 * std::sqrt(1.0 + R.get(0,0) - R.get(1,1) - R.get(2,2));
        w = (R.get(2,1) - R.get(1,2)) / s;
        x = 0.25 * s;
        y = (R.get(0,1) + R.get(1,0)) / s;
        z = (R.get(0,2) + R.get(2,0)) / s;
    } else if (R.get(1,1) > R.get(2,2)) {
        double s = 2.0 * std::sqrt(1.0 + R.get(1,1) - R.get(0,0) - R.get(2,2));
        w = (R.get(0,2) - R.get(2,0)) / s;
        x = (R.get(0,1) + R.get(1,0)) / s;
        y = 0.25 * s;
        z = (R.get(1,2) + R.get(2,1)) / s;
    } else {
        double s = 2.0 * std::sqrt(1.0 + R.get(2,2) - R.get(0,0) - R.get(1,1));
        w = (R.get(1,0) - R.get(0,1)) / s;
        x = (R.get(0,2) + R.get(2,0)) / s;
        y = (R.get(1,2) + R.get(2,1)) / s;
        z = 0.25 * s;
    }

    return Quaternion(w, x, y, z).normalize();
}

Quaternion Quaternion::fromVectors(const Vec3& from, const Vec3& to) {
    Vec3 f = from.normalize();
    Vec3 t = to.normalize();
    double d = f.dot(t);

    if (d > 0.999999) return identity();
    if (d < -0.999999) {
        // 180 degree rotation - find perpendicular axis
        Vec3 axis = Vec3::unitX().cross(f);
        if (axis.length() < 1e-6) axis = Vec3::unitY().cross(f);
        axis = axis.normalize();
        return {0, axis.x, axis.y, axis.z};
    }

    Vec3 c = f.cross(t);
    double w = 1.0 + d;
    return Quaternion(w, c.x, c.y, c.z).normalize();
}

Quaternion Quaternion::multiply(const Quaternion& q) const {
    return {
        w*q.w - x*q.x - y*q.y - z*q.z,
        w*q.x + x*q.w + y*q.z - z*q.y,
        w*q.y - x*q.z + y*q.w + z*q.x,
        w*q.z + x*q.y - y*q.x + z*q.w
    };
}

Vec3 Quaternion::rotate(const Vec3& v) const {
    // q * [0,v] * q*
    Quaternion p(0, v.x, v.y, v.z);
    Quaternion result = multiply(p).multiply(conjugate());
    return {result.x, result.y, result.z};
}

Vec3 Quaternion::inverseRotate(const Vec3& v) const {
    return conjugate().rotate(v);
}

Mat3 Quaternion::toRotationMatrix() const {
    double xx = x*x, yy = y*y, zz = z*z;
    double xy = x*y, xz = x*z, yz = y*z;
    double wx = w*x, wy = w*y, wz = w*z;

    Mat3 R;
    R.set(0, 0, 1 - 2*(yy + zz));
    R.set(0, 1, 2*(xy - wz));
    R.set(0, 2, 2*(xz + wy));
    R.set(1, 0, 2*(xy + wz));
    R.set(1, 1, 1 - 2*(xx + zz));
    R.set(1, 2, 2*(yz - wx));
    R.set(2, 0, 2*(xz - wy));
    R.set(2, 1, 2*(yz + wx));
    R.set(2, 2, 1 - 2*(xx + yy));
    return R;
}

Quaternion Quaternion::inverse() const {
    double n2 = w*w + x*x + y*y + z*z;
    if (n2 < 1e-14) return identity();
    double inv = 1.0 / n2;
    return {w * inv, -x * inv, -y * inv, -z * inv};
}

Quaternion Quaternion::normalize() const {
    double n = norm();
    if (n < 1e-14) return identity();
    double inv = 1.0 / n;
    return {w * inv, x * inv, y * inv, z * inv};
}

Quaternion Quaternion::derivative(const Vec3& omega) const {
    // q̇ = 0.5 * [0, ω_world] ⊗ q   (world-frame angular velocity, left multiply)
    Quaternion omegaQ(0, omega.x, omega.y, omega.z);
    Quaternion result = omegaQ.multiply(*this);
    return {result.w * 0.5, result.x * 0.5, result.y * 0.5, result.z * 0.5};
}

MatrixN Quaternion::getGMatrix() const {
    // G(q): 4×3 matrix that maps ω → q̇ via q̇ = 0.5 * G * ω
    MatrixN G(4, 3);
    G.set(0, 0, -x); G.set(0, 1, -y); G.set(0, 2, -z);
    G.set(1, 0,  w); G.set(1, 1, -z); G.set(1, 2,  y);
    G.set(2, 0,  z); G.set(2, 1,  w); G.set(2, 2, -x);
    G.set(3, 0, -y); G.set(3, 1,  x); G.set(3, 2,  w);
    return G.scale(0.5);
}

MatrixN Quaternion::getLMatrix() const {
    // Left multiplication matrix: q ⊗ p = L(q) * p
    MatrixN L(4, 4);
    L.set(0, 0,  w); L.set(0, 1, -x); L.set(0, 2, -y); L.set(0, 3, -z);
    L.set(1, 0,  x); L.set(1, 1,  w); L.set(1, 2, -z); L.set(1, 3,  y);
    L.set(2, 0,  y); L.set(2, 1,  z); L.set(2, 2,  w); L.set(2, 3, -x);
    L.set(3, 0,  z); L.set(3, 1, -y); L.set(3, 2,  x); L.set(3, 3,  w);
    return L;
}

MatrixN Quaternion::getRMatrix() const {
    // Right multiplication matrix: p ⊗ q = R(q) * p
    MatrixN R(4, 4);
    R.set(0, 0,  w); R.set(0, 1, -x); R.set(0, 2, -y); R.set(0, 3, -z);
    R.set(1, 0,  x); R.set(1, 1,  w); R.set(1, 2,  z); R.set(1, 3, -y);
    R.set(2, 0,  y); R.set(2, 1, -z); R.set(2, 2,  w); R.set(2, 3,  x);
    R.set(3, 0,  z); R.set(3, 1,  y); R.set(3, 2, -x); R.set(3, 3,  w);
    return R;
}

Quaternion Quaternion::slerp(const Quaternion& a, const Quaternion& b, double t) {
    double d = a.dot(b);
    Quaternion b2 = b;
    if (d < 0) {
        b2 = {-b.w, -b.x, -b.y, -b.z};
        d = -d;
    }

    if (d > 0.9995) {
        // Linear interpolation for very close quaternions
        return Quaternion(
            a.w + (b2.w - a.w) * t,
            a.x + (b2.x - a.x) * t,
            a.y + (b2.y - a.y) * t,
            a.z + (b2.z - a.z) * t
        ).normalize();
    }

    double theta = std::acos(d);
    double sinTheta = std::sin(theta);
    double wa = std::sin((1 - t) * theta) / sinTheta;
    double wb = std::sin(t * theta) / sinTheta;

    return Quaternion(
        wa * a.w + wb * b2.w,
        wa * a.x + wb * b2.x,
        wa * a.y + wb * b2.y,
        wa * a.z + wb * b2.z
    ).normalize();
}

} // namespace mb
