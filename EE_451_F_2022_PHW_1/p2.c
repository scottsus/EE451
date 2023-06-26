#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#define h 800 
#define w 800

#define input_file  "input.raw"
#define output_file "output.raw"

int abs(int);
int findCluster(int,int,int,int,int);

int main(int argc, char** argv){
    int i;
	struct timespec start, stop;
    FILE *fp;

  	unsigned char *a = (unsigned char*) malloc (sizeof(unsigned char)*h*w);
    
	// the matrix is stored in a linear array in row major fashion
	if (!(fp=fopen(input_file, "rb"))) {
		printf("can not open file\n");
		return 1;
	}
	fread(a, sizeof(unsigned char), w*h, fp);
	fclose(fp);
    
	// measure the start time here
	if ( clock_gettime( CLOCK_REALTIME, &start) == - 1) { perror("clock gettime");}
	
	// Your code goes here
	// Updating cluster mean values
	int k1=0, k2=85, k3=170, k4=255;
	for (int iter=0; iter<30; iter++) {
		int sum1=k1, sum2=k2, sum3=k3, sum4=k4;
		int num1=1, num2=1, num3=1, num4=1;
		for (i=0; i<h*w; i++) {
			int num = a[i];
			int cluster = findCluster(num, k1, k2, k3, k4);
			if (cluster == k1) {
				sum1 += num;
				num1++;
			} else if (cluster == k2) {
				sum2 += num;
				num2++;
			} else if (cluster == k3) {
				sum3 += num;
				num3++;
			} else if (cluster == k4) {
				sum4 += num;
				num4++;
			} else {
				printf("Cluster not found");
				return 0;
			}
		}
		k1 = sum1/num1;
		k2 = sum2/num2;
		k3 = sum3/num3;
		k4 = sum4/num4;
	}
	// Assigning mean values to clusters
	for (i=0; i<h*w; i++) {
		int num = a[i];
		int cluster = findCluster(num, k1, k2, k3, k4);
		if (cluster == k1) {
			a[i] = k1;
		} else if (cluster == k2) {
			a[i] = k2;
		} else if (cluster == k3) {
			a[i] = k3;
		} else if (cluster == k4) {
			a[i] = k4;
		} else {
			printf("Last step cluster not found");
			return 0;
		}
	}
	//
	
	// measure the end time here
	if ( clock_gettime( CLOCK_REALTIME, &stop) == -1) { perror("clock gettime");}
	double time = (stop.tv_sec - start.tv_sec) + (double)(stop.tv_nsec - start.tv_nsec)/1e9;
	
	// print out the execution time here
	printf("Number of FLOPs = %lu, Execution Time = %f sec, \n%lf MFLOPs per sec \n", (long)2*h*w*w, time, 1/time/1e6*2*h*w*w);
	
	if (!(fp=fopen(output_file,"wb"))) {
		printf("can not open file\n");
		return 1;
	}	
	fwrite(a, sizeof(unsigned char),w*h, fp);
    fclose(fp);
    
    return 0;
}

int abs(int x) {
	if (x < 0) { return -x; }
	return x;
}

int findCluster(int x, int k1, int k2, int k3, int k4) {
	int min = 99999, cluster;
	int diff = abs(x - k1);
	if (diff < min) { 
		min = diff;
		cluster = k1;
	}
	diff = abs(x - k2);
	if (diff < min) { 
		min = diff; 
		cluster = k2;
	}
	diff = abs(x - k3);
	if (diff < min) { 
		min = diff;
		cluster = k3;
	}
	diff = abs(x - k4);
	if (diff < min) { 
		min = diff;
		cluster = k4;
	}
	return cluster;
}