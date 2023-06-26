#include <stdlib.h>
#include <stdio.h>
#include <time.h>

int main(int argc, char* argv[]){
		int i, j, k;
		struct timespec start, stop;
		double time;
		int n = 2048; // matrix size is n*n
        int b;
        if (argc < 2) { 
            printf("CLI error: expected 2 args, got %d \n", argc);
            return 0;
        }
        b = atoi(argv[1]);
        int m = n/b;
		
		double **A = (double**) malloc (sizeof(double*)*n);
		double **B = (double**) malloc (sizeof(double*)*n);
		double **C = (double**) malloc (sizeof(double*)*n);
		for (i=0; i<n; i++) {
			A[i] = (double*) malloc(sizeof(double)*n);
			B[i] = (double*) malloc(sizeof(double)*n);
			C[i] = (double*) malloc(sizeof(double)*n);
		}
		
		for (i=0; i<n; i++){
			for(j=0; j<n; j++){
				A[i][j]=i;
				B[i][j]=i+j;
				C[i][j]=0;			
			}
		}
				
		if( clock_gettime(CLOCK_REALTIME, &start) == -1) { perror("clock gettime");}
		
		// Your code goes here //
		// Matrix C = Matrix A * Matrix B //	
		//*******************************//
        int bi, bj, bk;
		for (bi=0; bi<n; bi+=b)
            for (bj=0; bj<n; bj+=b)
                for (bk=0; bk<n; bk+=b)
                    for (i=0; i<b; i++)
                        for (j=0; j<b; j++)
                            for (k=0; k<b; k++)
                                C[bi+i][bj+j] += A[bi+i][bk+k] * B[bk+k][bj+j];
		//*******************************//
		
		if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror("clock gettime");}		
		time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
		
		printf("Number of FLOPs = %lu, Execution time = %f sec,\n%lf MFLOPs per sec\n", (long)2*n*n*n, time, 1/time/1e6*2*n*n*n);		
		printf("C[100][100]=%f\n", C[100][100]);
		
		// release memory
		for (i=0; i<n; i++) {
			free(A[i]);
			free(B[i]);
			free(C[i]);
		}
		free(A);
		free(B);
		free(C);
		return 0;
}
