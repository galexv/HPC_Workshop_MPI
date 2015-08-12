#include <stdio.h>
#include <math.h>

// #include <mpi.h>

/// Computes exp(x)
double fancy_exp(double x)
{
  double c=1;
  double s=c;

  for (unsigned long k=1; fabs(c)>2.E-16; ++k) {
    c *= x/k;
    s += c;
  }
  return s;
}

/// Computes log(x)
double fancy_log(double x)
{
  double c=x-1;
  double s=c;

  for (unsigned long k=0; fabs(c)>2.E-10; ++k) {
    c *= (1-x)*(k+1)/(k+2);
    s += c;
  }
  return s;
}

/// Computes pow(x,y)
double fancy_pow(double x, double y)
{
  double r=fancy_exp(y*fancy_log(x));
  return r;
}


/// Computes x**(-0.75)
double integrand(const double x)
{
  return fancy_pow(x,-0.75);
}

/// Computes \int_x1^x2 t^(-0.75) dt
double integral(const unsigned long npoints, const double x1, const double x2)
{
  double s=0;
  const double h=(x2-x1)/npoints;
#pragma omp parallel for schedule(runtime) reduction(+:s)  
  for (unsigned long i=0; i<npoints; ++i) {
    const double t=x1+(i+0.5)*h;
    const double y=integrand(t);
    s+=y;
  }
  s*=h;
  return s;
}

int main(int argc, char** argv)
{
  unsigned long int n;
  if (argc!=2 || sscanf(argv[1],"%lu",&n)!=1)  return 1;

  {
    double x=0.01;
    printf("Sanity check: logarithm accuracy at %lf is %lf\n",
           x, fabs(log(x)-fancy_log(x)));
  }
  
  printf("#x0 x I(x0,x) exact diff\n");
  double x0=0.00001;
  double x=1;
  const double y_exact=4*(pow(x,0.25)-pow(x0,0.25));
  const double y=integral(n, x0, x);
  printf("%lf %lf %lf %lf %lf\n", x0, x, y, y_exact, y-y_exact);

  return 0;
}
