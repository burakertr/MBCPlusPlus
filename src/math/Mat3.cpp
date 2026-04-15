#include "mb/math/Mat3.h"
#include <cmath>

namespace mb {

Mat3 Mat3::fromRotationX(double angle) {
    double c = std::cos(angle), s = std::sin(angle);
    Mat3 m;
    m.data.fill(0);
    m.set(0, 0, 1);
    m.set(1, 1, c); m.set(1, 2, -s);
    m.set(2, 1, s); m.set(2, 2, c);
    return m;
}

Mat3 Mat3::fromRotationY(double angle) {
    double c = std::cos(angle), s = std::sin(angle);
    Mat3 m;
    m.data.fill(0);
    m.set(0, 0, c); m.set(0, 2, s);
    m.set(1, 1, 1);
    m.set(2, 0, -s); m.set(2, 2, c);
    return m;
}

Mat3 Mat3::fromRotationZ(double angle) {
    double c = std::cos(angle), s = std::sin(angle);
    Mat3 m;
    m.data.fill(0);
    m.set(0, 0, c); m.set(0, 1, -s);
    m.set(1, 0, s); m.set(1, 1, c);
    m.set(2, 2, 1);
    return m;
}

Mat3 Mat3::multiply(const Mat3& other) const {
    Mat3 result;
    result.data.fill(0);
    for (int col = 0; col < 3; col++) {
        for (int row = 0; row < 3; row++) {
            double sum = 0;
            for (int k = 0; k < 3; k++) {
                sum += get(row, k) * other.get(k, col);
            }
            result.set(row, col, sum);
        }
    }
    return result;
}

Mat3 Mat3::transpose() const {
    Mat3 result;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            result.set(i, j, get(j, i));
    return result;
}

double Mat3::determinant() const {
    return get(0,0) * (get(1,1)*get(2,2) - get(1,2)*get(2,1))
         - get(0,1) * (get(1,0)*get(2,2) - get(1,2)*get(2,0))
         + get(0,2) * (get(1,0)*get(2,1) - get(1,1)*get(2,0));
}

Mat3 Mat3::inverse() const {
    double det = determinant();
    if (std::abs(det) < 1e-14) return identity();
    double invDet = 1.0 / det;

    Mat3 result;
    result.set(0, 0,  (get(1,1)*get(2,2) - get(1,2)*get(2,1)) * invDet);
    result.set(0, 1, -(get(0,1)*get(2,2) - get(0,2)*get(2,1)) * invDet);
    result.set(0, 2,  (get(0,1)*get(1,2) - get(0,2)*get(1,1)) * invDet);
    result.set(1, 0, -(get(1,0)*get(2,2) - get(1,2)*get(2,0)) * invDet);
    result.set(1, 1,  (get(0,0)*get(2,2) - get(0,2)*get(2,0)) * invDet);
    result.set(1, 2, -(get(0,0)*get(1,2) - get(0,2)*get(1,0)) * invDet);
    result.set(2, 0,  (get(1,0)*get(2,1) - get(1,1)*get(2,0)) * invDet);
    result.set(2, 1, -(get(0,0)*get(2,1) - get(0,1)*get(2,0)) * invDet);
    result.set(2, 2,  (get(0,0)*get(1,1) - get(0,1)*get(1,0)) * invDet);
    return result;
}

Mat3 Mat3::add(const Mat3& other) const {
    Mat3 result;
    for (int i = 0; i < 9; i++) result.data[i] = data[i] + other.data[i];
    return result;
}

Mat3 Mat3::sub(const Mat3& other) const {
    Mat3 result;
    for (int i = 0; i < 9; i++) result.data[i] = data[i] - other.data[i];
    return result;
}

Mat3 Mat3::scale(double s) const {
    Mat3 result;
    for (int i = 0; i < 9; i++) result.data[i] = data[i] * s;
    return result;
}

} // namespace mb
