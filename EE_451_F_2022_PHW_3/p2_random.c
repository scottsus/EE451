#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>

#define		size	   2*1024*1024

void swap(int*, int*);
int partition(int*, int, int);
void quickSort(int*, int, int);

int main(void){
	int i, j, tmp;
	struct timespec start, stop; 
	double exe_time;
	srand(time(NULL)); 
	int * m = (int *) malloc (sizeof(int)*size);
	for(i=0; i<size; i++){
		// m[i]=size-i;
		m[i] = rand();
	}
	
	if( clock_gettime(CLOCK_REALTIME, &start) == -1) { perror("clock gettime");}
	////////**********Your code goes here***************//
    int p=partition(m, 0, size);

    #pragma omp parallel shared(m, p)
    {
        #pragma omp sections nowait
        {
            #pragma omp section
            quickSort(m, 0, p);

            #pragma omp section
            quickSort(m, p+1, size);
        }
    }	
	///////******************************////
	
	if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror("clock gettime");}		
	exe_time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
	
	for(i=0;i<16;i++) printf("%d ", m[i]);		
	printf("\nExecution time = %f sec\n",  exe_time);		
}

void quickSort(int *arr, int start, int end){
	// you quick sort function goes here
	if (start>=end) return;
	int p=partition(arr, start, end);
	quickSort(arr, start, p);
	quickSort(arr, p+1, end);
}

int partition(int *arr, int start, int end) {
    int n=end-start;
    int pivotIdx=start+rand()%n, pivot=arr[pivotIdx];
    swap(&arr[pivotIdx], &arr[end-1]);
    int i, high=start;
    for (i=start; i<end-1; i++) {
        if (arr[i]<=pivot)
            swap(&arr[i], &arr[high++]);
    }
    swap(&arr[high], &arr[end-1]);
    return high;
}

void swap(int *a, int *b) {
	int temp=*a;
	*a=*b;
	*b=temp;
}
