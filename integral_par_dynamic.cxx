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


/// Calculates the fair share of `nsteps_all` steps between `nprocs` processes for rank `rank`
void get_steps(unsigned long nsteps_all, int nprocs, unsigned int rank,
               unsigned long* my_stepbase, unsigned long* my_nsteps)
{
  const unsigned long ns_share=nsteps_all/nprocs;
  const unsigned long ns_extra=nsteps_all%nprocs;
  if (rank<ns_extra) {
    *my_nsteps=ns_share+1;
    *my_stepbase=(ns_share+1)*rank;
  } else {
    *my_nsteps=ns_share;
    *my_stepbase=(ns_share+1)*ns_extra + (rank-ns_extra)*ns_share;
  }
  return;
}

/// Tags for master and workers actions
enum { 
  TAG_GO=1, TAG_STOP=2, TAG_READY=3, TAG_DONE=4
};


/// Worker process
void do_worker(int master, int rank)
{
  for (;;) {
    // Inform master that i am ready:
    MPI_Send(NULL,0, MPI_DOUBLE, master, TAG_READY, MPI_COMM_WORLD);
    // ...and wait for the task.
    double range_data[3]; // expect x1, x2 and the number of steps (as double!) 
    MPI_Status stat;
    MPI_Recv(range_data,3,MPI_DOUBLE, master, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
    switch (stat.MPI_TAG) {
    case TAG_GO:
      break; // do normal work
    case TAG_STOP:
      fprintf(stderr,"DEBUG: Rank %d: Got TAG_STOP.\n",rank);
      return;
    default:
      fprintf(stderr,"Rank %d: Got unexpected tag=%d, aborting.\n",rank,stat.MPI_TAG);
      MPI_Abort(MPI_COMM_WORLD, 2);
    }
    // we are here on TAG_GO.
    const double x1=range_data[0];
    const double x2=range_data[1];
    const unsigned long my_nsteps=range_data[2];
    fprintf(stderr,"DEBUG: Rank %u: %lu steps from %lf to %lf\n",
                   rank, my_nsteps, x1, x2);
    // Compute my own part of the integral
    double my_y=integral(my_nsteps, x1, x2);
    fprintf(stderr,"DEBUG: Rank %u: done, result=%lf\n",
                   rank, my_y);

    // Send the result to master
    MPI_Send(&my_y,1,MPI_DOUBLE, master, TAG_DONE, MPI_COMM_WORLD);
  }
}

/// Master process, Returns the computed result
double do_master(const double global_a, const double global_b,
               const unsigned long nsteps_all,
               const int nprocs, const int rank)
{
  const unsigned long points_per_block=1000000; // adjustable

  const double per_step=(global_b-global_a)/nsteps_all;
  
  int nworkers_left=nprocs-1;
  unsigned long ipoint=0; // next point to be processed
  double y=0;
  for (;;) {
    fprintf(stderr,"DEBUG: waiting for a message from a worker...\n");
    
    // Get a tagged message and possibly a result from any worker
    double y_worker=0;
    MPI_Status stat;
    MPI_Recv(&y_worker,1,MPI_DOUBLE, MPI_ANY_SOURCE,MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
    const int rank_worker=stat.MPI_SOURCE;
    switch (stat.MPI_TAG) {
    case TAG_READY:
      // Do we have any work for this worker?
      if (ipoint>=nsteps_all) {
        // if not, stop the worker
        fprintf(stderr,"DEBUG: Stopping worker %d\n",rank_worker);
        MPI_Send(NULL,0,MPI_DOUBLE, rank_worker, TAG_STOP, MPI_COMM_WORLD);
        --nworkers_left;
        break;
      }
      // Prepare chunk of work for the worker
      {
        const unsigned long ns_worker= (ipoint+points_per_block > nsteps_all)? (nsteps_all - ipoint) : points_per_block;
        const double x1=global_a+ipoint*per_step;
        const double x2=x1+ns_worker*per_step;
        double range_data[3];
        range_data[0]=x1;
        range_data[1]=x2;
        range_data[2]=ns_worker;
        // Send the chunk
        MPI_Send(range_data,3,MPI_DOUBLE, rank_worker, TAG_GO, MPI_COMM_WORLD);
        // adjust the amount of points
        ipoint += ns_worker;
      }
      break;

    case TAG_DONE:
      fprintf(stderr,"DEBUG: Received result from rank %d: %lf\n", rank_worker, y_worker);
      y += y_worker;
      break;

    default:
      fprintf(stderr,"Rank %d (master): Got unexpected tag=%d from %d, aborting.\n",rank,rank_worker,stat.MPI_TAG);
      MPI_Abort(MPI_COMM_WORLD,1);
    }

    // Any active workers left?
    if (nworkers_left<=0) {
      fprintf(stderr,"DEBUG: No more workers left, exiting the loop.\n");
      return y;
    }
  }
}



int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  // Get command line arguments, broadcast
  unsigned long int nsteps_all;
  if (rank==0) {
    if (argc!=2 || sscanf(argv[1],"%lu",&nsteps_all)!=1) {
      fprintf(stderr,"Usage:\n%s integration_steps\n\n\n",argv[0]);

      MPI_Abort(MPI_COMM_WORLD, 1);
      return 1;
    }

    if (nprocs<2) {
      fprintf(stderr,"At least 2 MPI processes must be running.\n\n\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
      return 1;
    }
    
    // Sanity check for reasonable convergence thresholds
    double x=0.01;
    printf("Sanity check: logarithm accuracy at %lf is %lf\n",
           x, fabs(log(x)-fancy_log(x)));
  }

  // MPI_Bcast(&nsteps_all, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

  // Global integration limits.
  const double global_a=1E-5;
  const double global_b=1;

  // Split into workers and master:
  if (rank==0) {
    // Run as the master and get the result:
    double y=do_master(global_a,global_b,nsteps_all,nprocs,rank);

    const double y_exact=4*(pow(global_b,0.25)-pow(global_a,0.25));
    printf("Result=%lf Exact=%lf Difference=%lf\n", y, y_exact, y-y_exact);

  } else {
    // Run as a worker 
    do_worker(0, rank);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  // Here we could start another computation.
  
  MPI_Finalize();
  return 0;
}
