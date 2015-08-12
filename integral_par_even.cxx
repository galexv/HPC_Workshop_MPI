#include <stdio.h>
#include <math.h>

#include <mpi.h>

/// Computes exp(x)
double fancy_exp(double x)
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
double fancy_log(double x)
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
  // #pragma omp parallel for schedule(runtime) reduction(+:s)  
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
  MPI_Init(&argc, &argv);

  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  // Get command line arguments, broadcast
  unsigned long int nsteps;
  if (rank==0) {
    if (argc!=2 || sscanf(argv[1],"%lu",&nsteps)!=1) {
      fprintf(stderr,"Usage: %s integration_steps_per_proc\n",argv[0]);

      MPI_Abort(MPI_COMM_WORLD, 1);
      return 1;
    }

    // Sanity check for reasonable convergence thresholds
    double x=0.01;
    printf("Sanity check: logarithm accuracy at %lf is %lf\n",
           x, fabs(log(x)-fancy_log(x)));
  }
  MPI_Bcast(&nsteps, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

  // Global integration limits.
  const double global_a=1E-5;
  const double global_b=1;
  
  // Each rank figures out its integration limits and number of steps
  const double range=(global_b-global_a)/nprocs;
  const double x1=global_a + rank*range;
  const double x2=global_a + (rank+1)*range;

  fprintf(stderr,"DEBUG: Rank %u: %lu steps from %lf to %lf\n",
                         rank, nsteps, x1, x2);

  // Compute my own part of the integral
  double my_y=integral(nsteps, x1, x2);
  fprintf(stderr,"DEBUG: Rank %u: done, result=%lf\n",
                 rank, my_y);

  // Sum all numbers on master
  double y=0;
  MPI_Reduce(&my_y, &y, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  // Print the results from the master
  if (rank==0) {
    const double y_exact=4*(pow(global_b,0.25)-pow(global_a,0.25));
    printf("Result=%lf Exact=%lf Difference=%lf\n", y, y_exact, y-y_exact);
  }

  MPI_Finalize();
  return 0;
}
