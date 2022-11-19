/* 
   Sample Solution to the game of life program using MPI.
   Author: Purushotham Bangalore
   Date: Feb 17, 2010

   Use -DDEBUG0 for printing local size, prev/next rank, counts, displs.
   Use -DDEBUG1 for output at the start and end.
   Use -DDEBUG2 for output at each iteration.

   To compile: gcc -Wall -O -o 2dgol 2dgol.c
   pgcc -fast -Minfo=accel -ta=tesla,cc70 -acc  2dgol.c -o 2dgol
   To run: 
   Local system: 
           ./2dgol  10 10
   
   -- To store the file in the scratch, you need to create a new folder (use your username), use this command: 
   mkdir /scratch/ualclsb00##
   To check the output file, use this command:
   cd  /scratch/ualclsb00##
   To load the module, you can load openmpi without CUDA 
   DMC Cluster at ASC:
           Follow instructions to execute MPI programs on DMC cluster
*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
//#include <mpi.h>

#define DIES   0
#define ALIVE  1

/* function to measure time taken */
double gettime(void) {
  struct timeval tval;

  gettimeofday(&tval, NULL);

  return( (double)tval.tv_sec + (double)tval.tv_usec/1000000.0 );
}

/* allocate row-major two-dimensional array */
int **allocarray(int P, int Q) {
  int i, *p, **a;

  p = (int *)malloc(P*Q*sizeof(int));
  a = (int **)malloc(P*sizeof(int*));
  for (i = 0; i < P; i++)
    a[i] = &p[i*Q]; 

  return a;
}

/* free allocated memory */
void freearray(int **a) {
  free(&a[0][0]);
  free(a);
}

/* print arrays in 2D format */
void printArray(int **a, int M, int N, int k) {
  int i, j;
  printf("Life after %d iterations:\n", k) ;
  for (i = 0; i < M+2; i++) {
    for (j = 0; j< N+2; j++)
      printf("%d ", a[i][j]);
    printf("\n");
  }
  printf("\n");
}
void copyArray(int **a, int **b,int M, int N) {
  int i, j;
  //#pragma acc kernels
  for (i = 0; i < M+2; i++) {
    for (j = 0; j< N+2; j++)
       a[i][j]=b[i][j];
    
  }
  
}
int compareArray(int **a, int **b,int M, int N) {
  int i, j;
  
  for (i = 0; i < M+2; i++) {
    for (j = 0; j< N+2; j++)
       if(a[i][j]!=b[i][j]){
          printf("## Mismatched[%d][%d]\n",i,j);
          return 0;
       }
    
  }
  return 0;
}

/* write array to a file (including ghost cells) */
void writefile(int **a, int N, FILE *fptr) {
  int i, j;
  for (i = 0; i < N+2; i++) {
    for (j = 0; j< N+2; j++)
      fprintf(fptr, "%d ", a[i][j]);
    fprintf(fptr, "\n");
  }
}

// int compute_acc(int **life, int **temp,int M, int N,int NTIMES){
//     int i, j;
//     #pragma acc data copy(life[:M+2][:N+2]) create(temp[:M+2][:N+2])
//     for(int k=0;k<NTIMES;k++)
//     {
//         #pragma acc parallel 
//         for (i = 1; i < M+1; i++) {
//                #pragma acc  loop 
//                 for (j = 1; j < N+1; j++) {
//                 /* find out the value of the current cell */
//                int value = life[i-1][j-1] + life[i-1][j] + life[i-1][j+1]
//                         + life[i][j-1]                   + life[i][j+1]
//                         + life[i+1][j-1] + life[i+1][j] + life[i+1][j+1] ;
                            
//                 // if(i==1 && j==2)
//                 // printf("value of life[%d][%d]=%d\n",i,j,value);
//                 /* check if the cell dies or life is born */
//                 if (life[i][j]) { // cell was alive in the earlier iteration
//                     if (value < 2 || value > 3) {
//                     temp[i][j] = DIES ;
//                     //flag++; // value changed 
//                     }
//                     else // value must be 2 or 3, so no need to check explicitly
//                     temp[i][j] = ALIVE ; // no change
//                         } 
//                 else { // cell was dead in the earlier iteration
//                     if (value == 3) {
//                     temp[i][j] = ALIVE;
//                     //flag++; // value changed 
//                     }
//                     else
//                     temp[i][j] = DIES; // no change
//                 }
//                 }
//             }
            
//             //copyArray(life,temp,M,N);
//             #pragma acc parallel 
//              for (i = 0; i < M+2; i++) {
//                  #pragma acc  loop 
//                 for (j = 0; j< N+2; j++)
//                 life[i][j]=temp[i][j];   
//             }
//     }
//     return 0;
// }

/* update each cell based on old values */
int compute(int **life, int **temp, int M, int N) {
  int i, j, value, flag=0;
  
  for (i = 1; i < M+1; i++) {
    for (j = 1; j < N+1; j++) {
      /* find out the value of the current cell */
      value = life[i-1][j-1] + life[i-1][j] + life[i-1][j+1]
            + life[i][j-1]                   + life[i][j+1]
              + life[i+1][j-1] + life[i+1][j] + life[i+1][j+1] ;
                  
      // if(i==1 && j==2)
      // printf("value of life[%d][%d]=%d\n",i,j,value);
      /* check if the cell dies or life is born */
      if (life[i][j]) { // cell was alive in the earlier iteration
        if (value < 2 || value > 3) {
          temp[i][j] = DIES ;
          flag++; // value changed 
        }
        else // value must be 2 or 3, so no need to check explicitly
          temp[i][j] = ALIVE ; // no change
            } 
      else { // cell was dead in the earlier iteration
        if (value == 3) {
          temp[i][j] = ALIVE;
          flag++; // value changed 
        }
        else
          temp[i][j] = DIES; // no change
      }
    }
  }

  return flag;
}

/* function to exchange boundary rows */
// void exchange(int **life, int N, int N, int prev, int next) {
//   // implement this function
  
//   MPI_Recv(&life[N+1][0], N+2, MPI_INT, next, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
//   MPI_Send(&life[1][0], N+2, MPI_INT, prev, 0, MPI_COMM_WORLD);
  
  
  
//   MPI_Recv(&life[0][0], N+2, MPI_INT, prev, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
//   MPI_Send(&life[N][0], N+2, MPI_INT, next, 0, MPI_COMM_WORLD);
  
  
// }


int main(int argc, char **argv) {
  int N, NTIMES, **life=NULL, **temp=NULL;
 // int **life_acc=NULL,**temp_acc=NULL;
//   int **ptr ;
  int i, j, k,flag=1;
//   int flag=1, myflag, rank, size, prev, next;
  //int N, **life=NULL, *bufptr=NULL, *counts=NULL, *displs=NULL;
  double t1,t2;
  char filename[BUFSIZ];
   FILE *fptr=NULL;

//   MPI_Init(&argc, &argv);
//   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//   MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (argc != 4) {
    printf("Usage: %s <problem size> <max. iterations> <output dir>\n", 
	   argv[0]);
   // MPI_Abort(MPI_COMM_WORLD, -1);
  }

  N = atoi(argv[1]);
  NTIMES = atoi(argv[2]);
  printf("Problem Size N=%d in number of Iterations=%d\n",N,NTIMES);

//   /* Compute the rows per process */
//   N = N/size + ((rank < (N%size))?1:0);

//   /* Compute ranks of neighboring processes */
//   prev = rank - 1;
//   next = rank + 1;
//   if (rank == 0) prev = MPI_PROC_NULL;
//   if (rank == size-1) next = MPI_PROC_NULL;

// #ifdef DEBUG0
//   printf("[%d]: N = %d prev = %d next = %d\n", rank, N, prev, next);
// #endif 
  
  /* Allocate memory for local life array and temp array */
  life = allocarray(N+2,N+2);
  temp = allocarray(N+2,N+2);
  // life_acc = allocarray(N+2,N+2);
  // temp_acc = allocarray(N+2,N+2);
  
  /* Initialize the boundaries of the temp matrix */
  for (i = 0; i < N+2; i++) {
    temp[i][0] = temp[i][N+1] = DIES ;
    life[i][0] = life[i][N+1] = DIES ;

  }
  for (j = 0; j < N+2; j++) {
    temp[0][j] = temp[N+1][j] = DIES ;
    life[0][j] = life[N+1][j] = DIES ;
  }
  
//   if(rank==size-1){
//     for(i=0;i<N+2;i++){
//         life[N+1][i] = DIES ;
//     }  
//   }
//   if(rank==0){
//      for (i = 0; i < N+2; i++) {
//        life[0][i] = DIES ;
//      }
//   }
  
//   if (rank == 0) {
//     /* Allocate memory for full life array */
//     life = allocarray(N+2,N+2);

//     /* Initialize the boundaries of the life array */
//     for (i = 0; i < N+2; i++) {
//       life[0][i] = life[i][0] = life[N+1][i] = life[i][N+1] = DIES ;
      
//     }

    /* Initialize the life array */
    for (i = 1; i < N+1; i++) {
      srand48(54321|i);
      for (j = 1; j< N+1; j++)
        if (drand48() < 0.5) 
          life[i][j] = ALIVE ;
        else
          life[i][j] = DIES ;
    }

#ifdef DEBUG1
    /* Display the initialized life matrix */
    printArray(life, N, N, 0);
#endif
// copyArray(life_acc,life,N,N);
// copyArray(temp_acc,temp,N,N);
// printArray(life, N, N, 0);
// printArray(life_acc, N, N, 0);
 
 /* set starting address of send buffer */
    //bufptr = &life[1][0];
  //} 
    /* compute the count and displacement values to scatter the array */
    // counts = malloc(sizeof(int)*size);
    // displs = malloc(sizeof(int)*size);

    // for (i = 0; i < size; i++)
    //   counts[i] = ((N/size) + ((i < (N%size))?1:0))*(N+2);
    // displs[0] = 0;
    // for (i = 1; i < size; i++)
    //   displs[i] = displs[i-1] + counts[i-1];

// #ifdef DEBUG0
//     //for (i = 0; i < size; i++)
//       printf("rank=%d: counts[%d] = %d displs[%d] =  %d\n", 
//              rank, rank, counts[rank], rank, displs[rank]);
// #endif 

   

  t1 = gettime();
  /* distribute the life array using 1-D distribution across rows */
//   MPI_Scatterv(bufptr, counts, displs, MPI_INT, 
//                &life[1][0], N*(N+2), MPI_INT, 0, MPI_COMM_WORLD);

  /* Play the game of life for given number of iterations */
  for (k = 0; k < NTIMES && flag != 0; k++) {
    #ifdef DEBUG0
        if (rank == 0) printf("[%d]: starting iteration %d\n", rank, k);
    #endif 
    /* exchange boundary values */
    //exchange(life, N, N, prev, next);
     
    /* compute local array */
    
        flag = compute(life, temp, N, N);
    // else
    //    flag = compute(temp, life, N, N);

    /* compute global flag value */
  //  MPI_Allreduce(&myflag, &flag, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    /* copy the new values to the old array */
    // ptr = life;
    // life = temp;
    // temp = ptr;
    copyArray(life,temp,N,N);
  
#ifdef DEBUG2
    /* Print no. of cells alive after the current iteration */
    // printf("No. of cells whose value changed in iteration %d = %d\n",
	//    k+1,flag) ;
    //printf("rank=%d\n",rank);
    /* Display the life matrix */
//     if(k%2==0)
//         printArray(temp, N, N, k+1);
//    else
        printArray(life, N, N, k+1);
#endif
  }
 
  /* collect the local life array back into life array */
//   MPI_Gatherv(&life[1][0], N*(N+2), MPI_INT, 
// 	      bufptr, counts, displs, MPI_INT, 0, MPI_COMM_WORLD);

  t2 = gettime() - t1;
  printf("Time taken %f seconds for %d iterations\n", t2, k);
//   MPI_Reduce(&t2, &t1, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

// t1 = gettime();
//  compute_acc(life_acc, temp_acc,N,N,NTIMES);
// t2 = gettime() - t1;
// printf("Acc Time taken %f seconds for %d iterations\n", t2, k);
 // printArray(life_acc, N, N, k);

  //compareArray(life,life_acc,N,N);
 

#ifdef DEBUG1
  /* Display the life matrix after k iterations */
//   if(rank==0){
  printArray(life, N, N, k);
//   }
#endif


    /* open file to write output */
    sprintf(filename,"%s/gol_project_output.%d.%d.txt", argv[3], N, NTIMES);
    if ((fptr = fopen(filename, "w")) == NULL) {
        printf("Error opening file %s for writing\n", filename);
        perror("fopen");
    //   MPI_Abort(MPI_COMM_WORLD, -1);
    }

    /* Write the final array to output file */
    printf("Writing output to file: %s\n", filename);
    writefile(life, N, fptr);
    fclose(fptr);


  freearray(life);
  freearray(temp);

  //MPI_Finalize();

  return 0;
}


