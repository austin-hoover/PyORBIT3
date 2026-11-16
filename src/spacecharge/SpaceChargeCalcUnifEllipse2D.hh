#ifndef SC_SPACECHARGE_CALC_UNIFORM_ELLIPSE_2D_HH
#define SC_SPACECHARGE_CALC_UNIFORM_ELLIPSE_2D_HH

#include <vector>

#include "Bunch.hh"
#include "CppPyWrapper.hh"

class SpaceChargeCalcUnifEllipse2D : public OrbitUtils::CppPyWrapper
{

public:
    explicit SpaceChargeCalcUnifEllipse2D(int nEllipses = 1);

    void trackBunch(Bunch* bunch, double length);

    void bunchAnalysis(Bunch* bunch);

    void calculateField(double x, double y, double& ex, double& ey) const;

    int getNEllipses() const;

    void getEllipseParameters(int index, double& a, double& b, double& lineDensity) const;

    void getCenter(double& x, double& y) const;

    double getAngle() const;

private:
    struct Ellipse
    {
        double a;
        double b;
        double q;
        Ellipse() : a(0.0), b(0.0), q(0.0) {}
    };
    std::vector<Ellipse> ellipses;
    double xCenter;
    double yCenter;
    double angle;
    double cosAngle;
    double sinAngle;
};

#endif
