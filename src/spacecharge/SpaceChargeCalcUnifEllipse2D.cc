#include "SpaceChargeCalcUnifEllipse2D.hh"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "ParticleMacroSize.hh"
#include "orbit_mpi.hh"
#include "wrap_mpi_comm.hh"

SpaceChargeCalcUnifEllipse2D::SpaceChargeCalcUnifEllipse2D(int nEllipses)
    : CppPyWrapper(NULL), xCenter(0.), yCenter(0.), angle(0.), cosAngle(1.), sinAngle(0.)
{
    if (nEllipses < 1)
        throw std::invalid_argument("nEllipses must be positive");
    ellipses.resize(nEllipses);
}

void SpaceChargeCalcUnifEllipse2D::bunchAnalysis(Bunch* bunch) {
    std::fill(ellipses.begin(), ellipses.end(), Ellipse());
    xCenter = 0.0;
    yCenter = 0.0;
    angle = 0.0;
    sinAngle = 0.0;
    cosAngle = 1.0;

    bunch->compress();
    ParticleMacroSize* weights = NULL;
    if (bunch->hasParticleAttributes("macrosize"))
        weights = static_cast<ParticleMacroSize*>(bunch->getParticleAttributes("macrosize"));
    const int count = bunch->getSize();
    const auto weight = [bunch, weights](int i)
    { return weights ? weights->macrosize(i) : bunch->getMacroSize(); };
    auto comm = bunch->getMPI_Comm_Local()->comm;

    // All ranks participate, including empty ranks. Reject invalid source
    // data collectively so no rank can leave its peers inside a reduction.
    double local[5] = {0., 0., 0., 0., 0.}, global[5];
    for (int i = 0; i < count; ++i)
    {
        const double w = weight(i);
        if (!std::isfinite(w) || w < 0. || !std::isfinite(bunch->x(i)) ||
            !std::isfinite(bunch->y(i)) || !std::isfinite(bunch->z(i)))
        {
            local[3] = 1.;
            continue;
        }
        local[0] += w;
        local[1] += w * bunch->x(i);
        local[2] += w * bunch->y(i);
        local[4] += w * bunch->z(i);
    }
    ORBIT_MPI_Allreduce(local, global, 5, MPI_DOUBLE, MPI_SUM, comm);
    if (global[3] != 0.)
        throw std::invalid_argument(
            "source weights must be nonnegative and position coordinates finite");
    const double total = global[0];
    if (total == 0.)
        return;
    if (!std::isfinite(total) || !std::isfinite(global[1]) || !std::isfinite(global[2]) ||
        !std::isfinite(global[4]))
        throw std::invalid_argument("non-finite weighted transverse moments");
    xCenter = global[1] / total;
    yCenter = global[2] / total;

    const double zCenter = global[4] / total;

    // A second pass avoids subtracting large centroid-squared moments.
    double covLocal[4] = {0., 0., 0., 0.}, cov[4];
    for (int i = 0; i < count; ++i)
    {
        if (weight(i) == 0.)
            continue;
        const double x = bunch->x(i) - xCenter;
        const double y = bunch->y(i) - yCenter;
        covLocal[0] += weight(i) * x * x;
        covLocal[1] += weight(i) * x * y;
        covLocal[2] += weight(i) * y * y;
        const double z = bunch->z(i) - zCenter;
        covLocal[3] += weight(i) * z * z;
    }
    ORBIT_MPI_Allreduce(covLocal, cov, 4, MPI_DOUBLE, MPI_SUM, comm);
    for (int j = 0; j < 4; ++j)
        cov[j] /= total;
    const double bunchLength = std::sqrt(12.0 * cov[3]);
    if (!std::isfinite(bunchLength) || bunchLength <= 0.)
        throw std::invalid_argument("rms bunch length <= 0");
    angle = 0.5 * std::atan2(2. * cov[1], cov[0] - cov[2]);
    cosAngle = std::cos(angle);
    sinAngle = std::sin(angle);
    const double major = 0.5 * (cov[0] + cov[2] + std::hypot(cov[0] - cov[2], 2. * cov[1]));
    const double minor = (cov[0] * cov[2] - cov[1] * cov[1]) / major;
    if (!std::isfinite(major) || !std::isfinite(minor) || major <= 0. || minor <= 0.)
        throw std::invalid_argument("the transverse position covariance must be positive definite");
    const double a = 2. * std::sqrt(major), b = 2. * std::sqrt(minor);
    const int n = getNEllipses();
    if (n == 1)
    {
        ellipses[0].a = a;
        ellipses[0].b = b;
        ellipses[0].q = total / bunchLength;
        return;
    }

    const auto radius = [this, bunch, a, b](int i)
    {
        const double x = bunch->x(i) - xCenter, y = bunch->y(i) - yCenter;
        return std::hypot((cosAngle * x + sinAngle * y) / a, (-sinAngle * x + cosAngle * y) / b);
    };
    double maxLocal = 0., maxRadius;
    for (int i = 0; i < count; ++i)
        if (weight(i) > 0.)
            maxLocal = std::max(maxLocal, radius(i));
    ORBIT_MPI_Allreduce(&maxLocal, &maxRadius, 1, MPI_DOUBLE, MPI_MAX, comm);
    std::vector<double> shellLocal(n, 0.), shell(n);
    for (int i = 0; i < count; ++i)
    {
        if (weight(i) == 0.)
            continue;
        const int bin = std::min(n - 1, static_cast<int>((radius(i) / maxRadius) * n));
        shellLocal[bin] += weight(i);
    }
    ORBIT_MPI_Allreduce(shellLocal.data(), shell.data(), n, MPI_DOUBLE, MPI_SUM, comm);

    // Shell i spans radii i*dr to (i+1)*dr. Divide by its area
    // (common factors cancel), difference adjacent densities, then multiply
    // by the area of the full ellipse. This telescopes to total/bunchLength.
    for (int i = 0; i < n; ++i)
        shell[i] /= (2. * i + 1.);
    for (int i = 0; i < n; ++i)
    {
        const double scale = maxRadius * (i + 1.) / n;
        ellipses[i].a = a * scale;
        ellipses[i].b = b * scale;
        ellipses[i].q =
            (shell[i] - (i + 1 < n ? shell[i + 1] : 0.)) * (i + 1.) * (i + 1.) / bunchLength;
    }
}

void SpaceChargeCalcUnifEllipse2D::calculateField(double x, double y, double& ex, double& ey) const {
    const double u = cosAngle * x + sinAngle * y, v = -sinAngle * x + cosAngle * y;
    double eu = 0., ev = 0.;
    for (const auto& e : ellipses)
    {
        if (e.q == 0.)
            continue;
        double aa = e.a, bb = e.b;
        const double r = std::hypot(u / aa, v / bb);
        if (r > 1.)
        {
            // Positive confocal root: u^2/(a^2+t)+v^2/(b^2+t)=1.
            // Parzen, physics/0108040, Eqs. (17)-(19), normalized to the
            // -log(r) Green function used by PoissonSolverFFT2D.
            const double a2 = aa * aa, b2 = bb * bb;
            const double B = u * u + v * v - a2 - b2;
            const double D = std::hypot(u * u - v * v - a2 + b2, 2. * u * v);
            // Rationalize when B<0 to avoid cancellation near the surface.
            const double C = a2 * b2 * (r - 1.) * (r + 1.);
            const double t = B >= 0. ? 0.5 * (B + D) : 2. * C / (D - B);
            aa = std::sqrt(a2 + t);
            bb = std::sqrt(b2 + t);
        }
        eu += 2. * e.q * (u / aa) / (aa + bb);
        ev += 2. * e.q * (v / bb) / (aa + bb);
    }
    ex = cosAngle * eu - sinAngle * ev;
    ey = sinAngle * eu + cosAngle * ev;
}

void SpaceChargeCalcUnifEllipse2D::trackBunch(Bunch* bunch, double length) {
    if (!std::isfinite(length))
        throw std::invalid_argument("length must be finite");
    const double beta = bunch->getSyncPart()->getBeta();
    const double gamma = bunch->getSyncPart()->getGamma();
    if (!std::isfinite(beta) || beta <= 0. || !std::isfinite(gamma))
        throw std::invalid_argument(
            "the synchronous particle must have positive finite kinetic energy");
    bunchAnalysis(bunch);
    const double factor =
        2. * length * bunch->getClassicalRadius() / (beta * beta * gamma * gamma * gamma);
    for (int i = 0; i < bunch->getSize(); ++i)
    {
        double ex, ey;
        calculateField(bunch->x(i) - xCenter, bunch->y(i) - yCenter, ex, ey);
        bunch->xp(i) += factor * ex;
        bunch->yp(i) += factor * ey;
    }
}

int SpaceChargeCalcUnifEllipse2D::getNEllipses() const {
    return static_cast<int>(ellipses.size());
}

void SpaceChargeCalcUnifEllipse2D::getEllipseParameters(int index, double& a, double& b, double& q) const {
    if (index < 0 || index >= getNEllipses())
        throw std::out_of_range("ellipse index out of range");
    a = ellipses[index].a;
    b = ellipses[index].b;
    q = ellipses[index].q;
}

void SpaceChargeCalcUnifEllipse2D::getCenter(double& x, double& y) const {
    x = xCenter;
    y = yCenter;
}
double SpaceChargeCalcUnifEllipse2D::getAngle() const
{
    return angle;
}
