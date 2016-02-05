#include <stdio.h>
#include <math.h>

#include <mpi.h>

#include "mylib/mymath.hpp"


/// Tags for manager and workers actions
enum { 
  TAG_GO=1, TAG_STOP=2, TAG_READY=3, TAG_DONE=4
};


/// Worker process
void do_worker(int manager, int rank)
{
  for (;;) {
    // Inform manager that i am ready:
    MPI_Send(NULL,0, MPI_DOUBLE, manager, TAG_READY, MPI_COMM_WORLD);
    // ...and wait for the task.
    double range_data[3]; // expect x1, x2 and the number of steps (as double!) 
    MPI_Status stat;
    MPI_Recv(range_data,3,MPI_DOUBLE, manager, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
    switch (stat.MPI_TAG) {
    case TAG_GO:
      break; // do normal work
    case TAG_STOP:
      // fprintf(stderr,"DEBUG: Rank %d: Got TAG_STOP.\n",rank);
      return;
    default:
      fprintf(stderr,"Rank %d: Got unexpected tag=%d, aborting.\n",rank,stat.MPI_TAG);
      MPI_Abort(MPI_COMM_WORLD, 2);
    }
    // we are here on TAG_GO.
    const double x1=range_data[0];
    const double x2=range_data[1];
    const unsigned long my_nsteps=range_data[2];
    // fprintf(stderr,"DEBUG: Rank %u: %lu steps from %lf to %lf\n", rank, my_nsteps, x1, x2);
    // Compute my own part of the integral
    double my_y=integral(integrand, my_nsteps, x1, x2);
    // fprintf(stderr,"DEBUG: Rank %u: done, result=%lf\n", rank, my_y);

    // Send the result to manager
    MPI_Send(&my_y,1,MPI_DOUBLE, manager, TAG_DONE, MPI_COMM_WORLD);
  }
}

/// Manager process, Returns the computed result
double do_manager(const double global_a, const double global_b,
                 const unsigned long nsteps_all, const unsigned long points_per_block,
                 const int nprocs, const int rank)
{
  const double per_step=(global_b-global_a)/nsteps_all;
  
  int nworkers_left=nprocs-1;
  unsigned long ipoint=0; // next point to be processed
  double y=0;
  for (;;) {
    // fprintf(stderr,"DEBUG: waiting for a message from a worker...\n");
    
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
        // fprintf(stderr,"DEBUG: Stopping worker %d\n",rank_worker);
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
      // fprintf(stderr,"DEBUG: Received result from rank %d: %lf\n", rank_worker, y_worker);
      y += y_worker;
      break;

    default:
      fprintf(stderr,"Rank %d (manager): Got unexpected tag=%d from %d, aborting.\n",rank,rank_worker,stat.MPI_TAG);
      MPI_Abort(MPI_COMM_WORLD,1);
    }

    // Any active workers left?
    if (nworkers_left<=0) {
      // fprintf(stderr,"DEBUG: No more workers left, exiting the loop.\n");
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
  unsigned long int nsteps_all, points_per_block;
  if (rank==0) {
    if (argc!=3
        || sscanf(argv[1],"%lu",&nsteps_all)!=1
        || sscanf(argv[2],"%lu",&points_per_block)!=1) {
      
      fprintf(stderr,"Usage:\n%s integration_steps points_per_block\n\n\n",argv[0]);

      MPI_Abort(MPI_COMM_WORLD, 1);
      return 1;
    }

    if (nprocs<2) {
      fprintf(stderr,"At least 2 MPI processes must be running.\n\n\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
      return 1;
    }
  }

  // Global integration limits.
  const double global_a=1E-5;
  const double global_b=1;

  // Split into workers and manager:
  if (rank==0) {
    // Run as the manager and get the result:
    double y=do_manager(global_a,global_b,nsteps_all,points_per_block,nprocs,rank);

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
