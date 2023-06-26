#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

struct thread_data {
    int row,col,n,thread_size;
    double **A,**B,**C;
};

void *matrix_mult(void*);
void print(double**,int); // for debugging

int main(int argc, char* argv[]) {
    int i,j;
    struct timespec start,stop;
    double time;
    int n=2048;
    if (argc<2) {
        printf("CLI error: expected 2 args, got %d\n", argc);
        return 0;
    }
    int p=atoi(argv[1]);
    int thread_size=n/p;

    double **A=(double**) malloc(n*sizeof(double*));
    double **B=(double**) malloc(n*sizeof(double*));
    double **C=(double**) malloc(n*sizeof(double*));
    for (i=0; i<n; i++) {
        A[i]=(double*) malloc(n*sizeof(double));
        B[i]=(double*) malloc(n*sizeof(double));
        C[i]=(double*) malloc(n*sizeof(double));
    }
    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            A[i][j]=i;
            B[i][j]=i+j;
            C[i][j]=0;
        }
    }

    pthread_t **threads = (pthread_t**) malloc(p*sizeof(pthread_t*));
    struct thread_data **args = (struct thread_data**) malloc(p*sizeof(struct thread_data*));
    for (i=0; i<p; i++) {
        threads[i]=((pthread_t*) malloc(p*sizeof(pthread_t)));
        args[i]=((struct thread_data*) malloc(p*sizeof(struct thread_data)));
        for (j=0; j<p; j++) {
            args[i][j].row=i*thread_size;
            args[i][j].col=j*thread_size;
            args[i][j].n=n;
            args[i][j].thread_size=thread_size;
            args[i][j].A=A;
            args[i][j].B=B;
            args[i][j].C=C;
        }
    }

    if (clock_gettime(CLOCK_REALTIME, &start) == -1) { perror("clock gettime"); }
    for (i=0; i<p; i++) {
        for (j=0; j<p; j++) {
            int rc=pthread_create(&threads[i][j], NULL, matrix_mult, (void*) &args[i][j]);
            if (rc) {
                printf("ERROR; return code from pthread_create() is %d\n", rc);
                exit(-1);
            }
        }
    }

    for (i=0; i<p; i++) {
        for (j=0; j<p; j++) {
            pthread_join(threads[i][j], NULL);
        }
    }
    if (clock_gettime(CLOCK_REALTIME, &stop) == -1) { perror("clock gettime"); }
    time=(stop.tv_sec-start.tv_sec)+(double)(stop.tv_nsec-start.tv_nsec)/1e9;

    printf("C[100][100]: %f\n", C[100][100]);
    printf("Execution time: %f sec\n", time);
    return 0;
}

void* matrix_mult(void* arg) {
    struct thread_data *data=(struct thread_data*)arg;
    int i,j,k;
    for (i=data->row; i<data->row+data->thread_size; i++) {
        for (j=data->col; j<data->col+data->thread_size; j++) {
            for (k=0; k<data->n; k++) {
                data->C[i][j] += data->A[i][k] * data->B[k][j];
            }
        }
    }
    pthread_exit(NULL);
}

void print(double** M, int n) {
    int i,j;
    printf("Printing matrix:\n");
    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            printf("%f ",M[i][j]);
        }
        printf("\n");
    }
}
