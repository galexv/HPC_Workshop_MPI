#include <stdio.h>
#include <math.h>

// #include <mpi.h>

/// Computes x**0.5
double fancy_sqrt(const double x)
{
  // const double alpha=0.001;
  const double alpha=0.5;
  const double thresh=2.E-16;

  double y=1.0;
  for (;;) {
    double y_next = alpha*x/y + (1.-alpha)*y;
    if (fabs(y_next-y)<thresh) break;
    y=y_next;
  }

  return y;
}

/// Computes exp(x)
double fancy_exp(double x, unsigned long* niter)
{
  double c=1;
  double s=c;

  unsigned long k=1;
  for (; fabs(c)>2.E-16; ++k) {
    c *= x/k;
    s += c;
  }
  *niter=k;
  return s;
}

/// Computes log(x)
double fancy_log(double x, unsigned long* niter)
{
  double c=x-1;
  double s=c;

  unsigned long k=0;
  for (; fabs(c)>2.E-16; ++k) {
    c *= (1-x)*(k+1)/(k+2);
    s += c;
  }
  *niter=k;
  return s;
}

/// Computes pow(x,y)
double fancy_pow(double x, double y, unsigned long* niter)
{
  unsigned long n1, n2;
  double r=fancy_exp(y*fancy_log(x,&n1),&n2);
  *niter=n1+n2;
  return r;
}


/// Computes x**(-0.75)
double integrand(const double x, unsigned long* niter)
{
  // return fancy_sqrt(fancy_sqrt(1./(x*x*x)));
  // return pow(x,-0.5);
  // return pow(x,-0.75);
  return fancy_pow(x,-0.75,niter);
}

/// Computes \int_x1^x2 t^(-0.75) dt
double integral(const unsigned long npoints, const double x1, const double x2, unsigned long* niter)
{
  double s=0;
  const double h=(x2-x1)/npoints;
  unsigned long nit;
  *niter=0;
  for (unsigned long i=0; i<npoints; ++i) {
    const double t=x1+(i+0.5)*h;
    const double y=integrand(t,&nit);
    s+=y;
    *niter += nit;
  }
  s*=h;
  return s;
}

int main(int argc, char** argv)
{
  double s=0.001;
  unsigned long int n;
  if (argc!=2 || sscanf(argv[1],"%lu",&n)!=1)  return 1;
  printf("#x I(x,x+0.1) exact diff\n");
  for (double x=s; x<=1.0; x+=s) {
    double x1=x+s;
    const double y_exact=4*(pow(x1,0.25)-pow(x,0.25));
    unsigned long niter;
    const double y=integral(n, x, x1, &niter);
    printf("%lf %lf %lf %lf %lu\n", x, y, y_exact, y-y_exact, niter);
    fflush(stdout);
  }
  return 0;
}

