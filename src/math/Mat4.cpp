#include "mb/math/Mat4.h"
#include "mb/math/Quaternion.h"
#include <cmath>

namespace mb {

Mat4 Mat4::identity() {
    Mat4 m;
    m.data.fill(0);
    m.data[0] = m.data[5] = m.data[10] = m.data[15] = 1.0;
    return m;
}

Mat4 Mat4::zero() {
    Mat4 m;
    m.data.fill(0);
    return m;
}

Mat4 Mat4::fromRotationTranslation(const Mat3& R, const Vec3& t) {
    Mat4 m;
    m.data.fill(0);
    for (int c = 0; c < 3; c++)
        for (int r = 0; r < 3; r++)
            m.set(r, c, R.get(r, c));
    m.set(0, 3, t.x);
    m.set(1, 3, t.y);
    m.set(2, 3, t.z);
    m.set(3, 3, 1.0);
    return m;
}

Mat4 Mat4::fromQuaternionTranslation(const Quaternion& q, const Vec3& t) {
    return fromRotationTranslation(q.toRotationMatrix(), t);
}

Vec3 Mat4::transformPoint(const Vec3& p) const {
    return {
        get(0,0)*p.x + get(0,1)*p.y + get(0,2)*p.z + get(0,3),
        get(1,0)*p.x + get(1,1)*p.y + get(1,2)*p.z + get(1,3),
        get(2,0)*p.x + get(2,1)*p.y + get(2,2)*p.z + get(2,3)
    };
}

Vec3 Mat4::transformDirection(const Vec3& d) const {
    return {
        get(0,0)*d.x + get(0,1)*d.y + get(0,2)*d.z,
        get(1,0)*d.x + get(1,1)*d.y + get(1,2)*d.z,
        get(2,0)*d.x + get(2,1)*d.y + get(2,2)*d.z
    };
}

Mat3 Mat4::getRotation() const {
    Mat3 R;
    for (int c = 0; c < 3; c++)
        for (int r = 0; r < 3; r++)
            R.set(r, c, get(r, c));
    return R;
}

Vec3 Mat4::getTranslation() const {
    return {get(0, 3), get(1, 3), get(2, 3)};
}

Mat4 Mat4::multiply(const Mat4& other) const {
    Mat4 result;
    result.data.fill(0);
    for (int col = 0; col < 4; col++) {
        for (int row = 0; row < 4; row++) {
            double sum = 0;
            for (int k = 0; k < 4; k++)
                sum += get(row, k) * other.get(k, col);
            result.set(row, col, sum);
        }
    }
    return result;
}

Mat4 Mat4::inverseRigid() const {
    // For rigid body transform [R t; 0 1]: inverse = [R^T  -R^T*t; 0 1]
    Mat3 R = getRotation();
    Vec3 t = getTranslation();
    Mat3 RT = R.transpose();
    Vec3 invT = RT.multiplyVec3(t).negate();
    return fromRotationTranslation(RT, invT);
}

Mat4 Mat4::inverse() const {
    // Gauss-Jordan elimination
    double a[4][8];
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++)
            a[r][c] = get(r, c);
        for (int c = 0; c < 4; c++)
            a[r][4 + c] = (r == c) ? 1.0 : 0.0;
    }

    for (int col = 0; col < 4; col++) {
        // Find pivot
        int maxRow = col;
        double maxVal = std::abs(a[col][col]);
        for (int r = col + 1; r < 4; r++) {
            if (std::abs(a[r][col]) > maxVal) {
                maxVal = std::abs(a[r][col]);
                maxRow = r;
            }
        }
        if (maxVal < 1e-14) return identity();

        // Swap rows
        if (maxRow != col)
            for (int c = 0; c < 8; c++)
                std::swap(a[col][c], a[maxRow][c]);

        // Scale pivot row
        double pivot = a[col][col];
        for (int c = 0; c < 8; c++)
            a[col][c] /= pivot;

        // Eliminate column
        for (int r = 0; r < 4; r++) {
            if (r == col) continue;
            double factor = a[r][col];
            for (int c = 0; c < 8; c++)
                a[r][c] -= factor * a[col][c];
        }
    }

    Mat4 result;
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++)
            result.set(r, c, a[r][4 + c]);
    return result;
}

} // namespace mb
