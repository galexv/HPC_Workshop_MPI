/**
  \file libmath.cxx
  \brief Computational part of the demo program
*/

#include <math.h>

/// Computes exp(x)
static double fancy_exp(double x)
{
  double c=1;
  double s=c;
  const double thresh=2.E-16;

  for (unsigned long k=1; fabs(c)>thresh; ++k) {
    c *= x/k;
    s += c;
  }
  return s;
}

/// Computes log(x)
static double fancy_log(double x)
{
  double c=x-1;
  double s=c;
  const double thresh=2.E-10;

  for (unsigned long k=0; fabs(c)>thresh; ++k) {
    c *= (1-x)*(k+1)/(k+2);
    s += c;
  }
  return s;
}

/// Computes pow(x,y)
static double fancy_pow(double x, double y)
{
  double r=fancy_exp(y*fancy_log(x));
  return r;
}


/// Computes x**(-0.75)
double integrand(const double x)
{
  return fancy_pow(x,-0.75);
}

/// Computes integral of f(x) from x1 to x2 divided into npoints points
double integral(double (*f)(double), const unsigned long npoints, const double x1, const double x2)
{
  double s=0;
  const double h=(x2-x1)/npoints;
  // #pragma omp parallel for schedule(runtime) reduction(+:s)  
  for (unsigned long i=0; i<npoints; ++i) {
    const double t=x1+(i+0.5)*h;
    const double y=f(t);
    s+=y;
  }
  s*=h;
  return s;
}
