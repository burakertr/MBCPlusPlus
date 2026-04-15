#include "mb/math/Vec3.h"
#include "mb/math/Mat3.h"
#include <sstream>

namespace mb {

Mat3 Vec3::skewSymmetric() const {
    return Mat3::fromSkewSymmetric(*this);
}

std::string Vec3::toString() const {
    std::ostringstream ss;
    ss << "Vec3(" << x << ", " << y << ", " << z << ")";
    return ss.str();
}

} // namespace mb
