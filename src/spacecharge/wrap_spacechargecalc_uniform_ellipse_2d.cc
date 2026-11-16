#include "wrap_spacechargecalc_uniform_ellipse_2d.hh"

#include <cmath>
#include <new>
#include <stdexcept>

#include "SpaceChargeCalcUnifEllipse2D.hh"
#include "pyORBIT_Object.hh"
#include "wrap_bunch.hh"

namespace wrap_spacecharge
{
namespace
{
typedef SpaceChargeCalcUnifEllipse2D Calculator;

Calculator* calculator(PyObject* self)
{
    return static_cast<Calculator*>(reinterpret_cast<pyORBIT_Object*>(self)->cpp_obj);
}

PyObject* newCalculator(PyTypeObject* type, PyObject*, PyObject*)
{
    pyORBIT_Object* self = reinterpret_cast<pyORBIT_Object*>(type->tp_alloc(type, 0));
    if (!self)
        return NULL;
    self->cpp_obj = NULL;
    try
    {
        // Keep the object usable even when a Python subclass overrides __init__.
        self->cpp_obj = new Calculator();
    }
    catch (const std::bad_alloc&)
    {
        Py_DECREF(self);
        return PyErr_NoMemory();
    }
    return reinterpret_cast<PyObject*>(self);
}

int init(pyORBIT_Object* self, PyObject* args, PyObject* kwds)
{
    int n = 1;
    static const char* names[] = {"nEllipses", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|i:SpaceChargeCalcUnifEllipse2D",
                                     const_cast<char**>(names), &n))
        return -1;
    if (n < 1)
    {
        PyErr_SetString(PyExc_ValueError, "nEllipses must be positive");
        return -1;
    }
    try
    {
        Calculator* next = new Calculator(n);
        delete static_cast<Calculator*>(self->cpp_obj);
        self->cpp_obj = next;
        return 0;
    }
    catch (const std::invalid_argument& e)
    {
        PyErr_SetString(PyExc_ValueError, e.what());
    }
    catch (const std::bad_alloc&)
    {
        PyErr_NoMemory();
    }
    return -1;
}

void dealloc(pyORBIT_Object* self)
{
    delete static_cast<Calculator*>(self->cpp_obj);
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

Bunch* getBunch(PyObject* obj)
{
    const int valid = PyObject_IsInstance(obj, wrap_orbit_bunch::getBunchType("Bunch"));
    if (valid <= 0)
    {
        if (valid == 0)
            PyErr_SetString(PyExc_TypeError, "bunch must be a Bunch");
        return NULL;
    }
    return static_cast<Bunch*>(reinterpret_cast<pyORBIT_Object*>(obj)->cpp_obj);
}

PyObject* trackBunch(PyObject* self, PyObject* args)
{
    PyObject* obj;
    double length;
    if (!PyArg_ParseTuple(args, "Od:trackBunch", &obj, &length))
        return NULL;
    Bunch* bunch = getBunch(obj);
    if (!bunch)
        return NULL;
    try
    {
        calculator(self)->trackBunch(bunch, length);
        Py_RETURN_NONE;
    }
    catch (const std::invalid_argument& e)
    {
        PyErr_SetString(PyExc_ValueError, e.what());
    }
    catch (const std::bad_alloc&)
    {
        return PyErr_NoMemory();
    }
    return NULL;
}

PyObject* bunchAnalysis(PyObject* self, PyObject* args)
{
    PyObject* obj;
    if (!PyArg_ParseTuple(args, "O:bunchAnalysis", &obj))
        return NULL;
    Bunch* bunch = getBunch(obj);
    if (!bunch)
        return NULL;
    try
    {
        calculator(self)->bunchAnalysis(bunch);
        Py_RETURN_NONE;
    }
    catch (const std::invalid_argument& e)
    {
        PyErr_SetString(PyExc_ValueError, e.what());
    }
    catch (const std::bad_alloc&)
    {
        return PyErr_NoMemory();
    }
    return NULL;
}

PyObject* calculateField(PyObject* self, PyObject* args)
{
    double x, y, ex, ey;
    if (!PyArg_ParseTuple(args, "dd:calculateField", &x, &y))
        return NULL;
    if (!std::isfinite(x) || !std::isfinite(y))
    {
        PyErr_SetString(PyExc_ValueError, "field coordinates must be finite");
        return NULL;
    }
    calculator(self)->calculateField(x, y, ex, ey);
    return Py_BuildValue("(dd)", ex, ey);
}

PyObject* getNEllipses(PyObject* self, PyObject*)
{
    return PyLong_FromLong(calculator(self)->getNEllipses());
}

PyObject* getEllipseParameters(PyObject* self, PyObject* args)
{
    int i;
    double a, b, q;
    if (!PyArg_ParseTuple(args, "i:getEllipseParameters", &i))
        return NULL;
    try
    {
        calculator(self)->getEllipseParameters(i, a, b, q);
    }
    catch (const std::out_of_range& e)
    {
        PyErr_SetString(PyExc_IndexError, e.what());
        return NULL;
    }
    return Py_BuildValue("(ddd)", a, b, q);
}

PyObject* getCenter(PyObject* self, PyObject*)
{
    double x, y;
    calculator(self)->getCenter(x, y);
    return Py_BuildValue("(dd)", x, y);
}

PyObject* getAngle(PyObject* self, PyObject*)
{
    return PyFloat_FromDouble(calculator(self)->getAngle());
}

PyMethodDef methods[] = {
    {"trackBunch", trackBunch, METH_VARARGS,
     "trackBunch(bunch, length): apply a transverse kick; lengths in metres."},
    {"bunchAnalysis", bunchAnalysis, METH_VARARGS, "bunchAnalysis(bunch): fit without kicking."},
    {"calculateField", calculateField, METH_VARARGS,
     "calculateField(dx, dy): normalized field in lab axes, relative to the centroid."},
    {"getNEllipses", getNEllipses, METH_NOARGS, "Return the number of fitted ellipses."},
    {"getEllipseParameters", getEllipseParameters, METH_VARARGS,
     "Return (a, b, signed particle line density) in the principal frame."},
    {"getCenter", getCenter, METH_NOARGS, "Return the fitted transverse centroid (x, y)."},
    {"getAngle", getAngle, METH_NOARGS, "Return the major-axis angle from lab x in radians."},
    {NULL, NULL, 0, NULL}};

PyTypeObject type = {PyVarObject_HEAD_INIT(NULL, 0)};
} // namespace

void initSpaceChargeCalcUniformEllipse2D(PyObject* module)
{
    type.tp_name = "orbit.core.spacecharge.SpaceChargeCalcUnifEllipse2D";
    type.tp_basicsize = sizeof(pyORBIT_Object);
    type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    type.tp_doc = "Free-space uniform-ellipse transverse space charge solver; "
                  "SpaceChargeCalcUnifEllipse2D(nEllipses=1).";
    type.tp_new = newCalculator;
    type.tp_init = reinterpret_cast<initproc>(init);
    type.tp_dealloc = reinterpret_cast<destructor>(dealloc);
    type.tp_methods = methods;
    if (PyType_Ready(&type) < 0)
        return;
    Py_INCREF(&type);
    PyModule_AddObject(module, "SpaceChargeCalcUnifEllipse2D", reinterpret_cast<PyObject*>(&type));
}
} // namespace wrap_spacecharge
