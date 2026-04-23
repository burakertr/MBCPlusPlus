#include "mb/core/State.h"
#include <cmath>

namespace mb {

StateVector StateVector::fromBodies(const std::vector<Body*>& bodies, int numConstr) {
    StateVector s;
    s.numBodies = static_cast<int>(bodies.size());
    s.numConstraints = numConstr;
    s.totalNq = 0;
    s.totalNv = 0;

    for (auto* body : bodies) {
        s.qOffsets.push_back(s.totalNq);
        s.vOffsets.push_back(s.totalNv);
        s.nqPerBody.push_back(body->nq());
        s.nvPerBody.push_back(body->nv());
        s.dynamicBody.push_back(body->isDynamic());
        s.totalNq += body->nq();
        s.totalNv += body->nv();
    }

    s.q.resize(s.totalNq, 0.0);
    s.v.resize(s.totalNv, 0.0);
    s.a.resize(s.totalNv, 0.0);
    s.lambda.resize(numConstr, 0.0);

    // Copy initial state from bodies
    for (int i = 0; i < static_cast<int>(bodies.size()); i++) {
        s.copyFromBody(i, bodies[i]);
    }

    return s;
}

StateVector StateVector::clone() const {
    StateVector s;
    s.time = time;
    s.q = q;
    s.v = v;
    s.a = a;
    s.lambda = lambda;
    s.qOffsets = qOffsets;
    s.vOffsets = vOffsets;
    s.nqPerBody = nqPerBody;
    s.nvPerBody = nvPerBody;
    s.dynamicBody = dynamicBody;
    s.numBodies = numBodies;
    s.numConstraints = numConstraints;
    s.totalNq = totalNq;
    s.totalNv = totalNv;
    return s;
}

void StateVector::copyFrom(const StateVector& other) {
    time = other.time;
    q = other.q;
    v = other.v;
    a = other.a;
    lambda = other.lambda;
    qOffsets = other.qOffsets;
    vOffsets = other.vOffsets;
    nqPerBody = other.nqPerBody;
    nvPerBody = other.nvPerBody;
    dynamicBody = other.dynamicBody;
    numBodies = other.numBodies;
    numConstraints = other.numConstraints;
    totalNq = other.totalNq;
    totalNv = other.totalNv;
}

StateVector StateVector::addScaled(const StateVector& other, double scale) const {
    StateVector s = clone();
    s.time = time + other.time * scale;
    for (int i = 0; i < totalNq; i++)
        s.q[i] = q[i] + other.q[i] * scale;
    for (int i = 0; i < totalNv; i++)
        s.v[i] = v[i] + other.v[i] * scale;
    return s;
}

void StateVector::normalizeQuaternions() {
    for (int i = 0; i < numBodies; i++) {
        if (nqPerBody[i] == 7) {
            int off = qOffsets[i] + 3;
            double w = q[off], x = q[off+1], y = q[off+2], z = q[off+3];
            double n = std::sqrt(w*w + x*x + y*y + z*z);
            if (n > 1e-14) {
                double inv = 1.0 / n;
                q[off] *= inv; q[off+1] *= inv; q[off+2] *= inv; q[off+3] *= inv;
            }
        }
    }
}

std::vector<double> StateVector::computeQDot(const std::vector<Body*>& bodies) const {
    std::vector<double> qDot(totalNq, 0.0);
    for (int i = 0; i < numBodies; i++) {
        auto bodyQDot = bodies[i]->computeQDot();
        int off = qOffsets[i];
        for (int j = 0; j < nqPerBody[i]; j++) {
            qDot[off + j] = bodyQDot[j];
        }
    }
    return qDot;
}

void StateVector::copyToBody(int bodyIndex, Body* body) const {
    auto bodyQ = std::vector<double>(
        q.begin() + qOffsets[bodyIndex],
        q.begin() + qOffsets[bodyIndex] + nqPerBody[bodyIndex]
    );
    body->setQ(bodyQ);

    auto bodyV = std::vector<double>(
        v.begin() + vOffsets[bodyIndex],
        v.begin() + vOffsets[bodyIndex] + nvPerBody[bodyIndex]
    );
    body->setV(bodyV);
}

void StateVector::copyFromBody(int bodyIndex, Body* body) {
    auto bodyQ = body->getQ();
    int qOff = qOffsets[bodyIndex];
    for (int j = 0; j < nqPerBody[bodyIndex]; j++)
        q[qOff + j] = bodyQ[j];

    auto bodyV = body->getV();
    int vOff = vOffsets[bodyIndex];
    for (int j = 0; j < nvPerBody[bodyIndex]; j++)
        v[vOff + j] = bodyV[j];
}

Vec3 StateVector::getPosition(int bodyIndex) const {
    int off = qOffsets[bodyIndex];
    return Vec3(q[off], q[off+1], q[off+2]);
}

Vec3 StateVector::getVelocity(int bodyIndex) const {
    int off = vOffsets[bodyIndex];
    return Vec3(v[off], v[off+1], v[off+2]);
}

Quaternion StateVector::getOrientation(int bodyIndex) const {
    if (nqPerBody[bodyIndex] >= 7) {
        int off = qOffsets[bodyIndex] + 3;
        return Quaternion(q[off], q[off+1], q[off+2], q[off+3]);
    }
    return Quaternion::identity();
}

Vec3 StateVector::getAngularVelocity(int bodyIndex) const {
    if (nvPerBody[bodyIndex] >= 6) {
        int off = vOffsets[bodyIndex] + 3;
        return Vec3(v[off], v[off+1], v[off+2]);
    }
    return Vec3::zero();
}

} // namespace mb
