#include <stdio.h>
#include <math.h>

#include "mylib/mymath.hpp"

int main(int argc, char** argv)
{
  unsigned long int n;
  if (argc!=2 || sscanf(argv[1],"%lu",&n)!=1) {
    fprintf(stderr,"Usage:\n%s integration_steps\n\n\n",argv[0]);

  // Integration limits.
  const double global_a=1E-5;
  const double global_b=1;

  const double y=integral(integrand, n, global_a, global_b);

  const double y_exact=4*(pow(global_b,0.25)-pow(global_a,0.25));
  printf("Result=%lf Exact=%lf Difference=%lf\n", y, y_exact, y-y_exact);

  return 0;
}
