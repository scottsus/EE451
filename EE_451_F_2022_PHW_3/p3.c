#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <pthread.h>

#define h 800
#define w 800
#define num_clusters 6

#define input_file "input.raw"
#define output_file "output.raw"

void* update_cluster(void*);
int abs(int);
int find_cluster(int,int[]);

struct thread_data {
    int id;
    int *k, *sums, *nums;
    int start, end;
    unsigned char* a;
    pthread_mutex_t *mutex;
    pthread_cond_t *cond;
    int p, *r;
};

int main(int argc, char** argv) {
    int i;
    if (argc<2) {
        printf("CLI error: expected 2 args, got %d\n", argc);
        return 1;
    }
    int p=atoi(argv[1]);
    int thread_size=h*w/p;
    pthread_mutex_t mutex=PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t cond=PTHREAD_COND_INITIALIZER;
    struct timespec start, stop;
    FILE *fp;

    unsigned char *a = (unsigned char*) malloc(h*w*sizeof(unsigned char));

    if (!(fp=fopen(input_file, "rb"))) {
        printf("can't open input file\n");
        return 1;
    }
    fread(a, sizeof(unsigned char), h*w, fp);
    fclose(fp);

    if (clock_gettime(CLOCK_REALTIME, &start)==-1) perror("clock gettime");
    pthread_t *threads=(pthread_t*) malloc(p*sizeof(pthread_t));
    struct thread_data *args=(struct thread_data*) malloc(p*sizeof(struct thread_data));

    int r=0;
    int k[num_clusters]={0, 65, 100, 125, 190, 255};
    for (int iter=0; iter<50; iter++) {
        int sums[6]={k[0], k[1], k[2], k[3], k[4], k[5]};
        int nums[6]={1, 1, 1, 1, 1, 1};
        for (i=0; i<p; i++) {
            args[i].id=i;
            args[i].k=k;
            args[i].sums=sums;
            args[i].nums=nums;
            args[i].start=i*thread_size;
            args[i].end=i*thread_size+thread_size;
            args[i].a=a;
            args[i].mutex=&mutex;
            args[i].cond=&cond;
            args[i].p=p;
            args[i].r=&r;
        }
        for (i=0; i<p; i++) {
            int rc=pthread_create(&threads[i], NULL, update_cluster, (void*) &args[i]);
            if (rc) {
                printf("ERROR; return code from pthread_create() is %d\n", rc);
                exit(-1);
            }
        }
        pthread_mutex_lock(&mutex);
        if (r==p)
            pthread_cond_broadcast(&cond);
        else
            pthread_cond_wait(&cond, &mutex);
        r=0;
        pthread_mutex_unlock(&mutex);
    }

    for (i=0; i<p; i++) {
        pthread_join(threads[i], NULL);
    }

    for (i=0; i<num_clusters; i++) {
        printf("%d: %d\n", i, k[i]);
    }

    for (i=0; i<h*w; i++) {
        int cluster=find_cluster(a[i], k);
        for (int j=0; j<num_clusters; j++) {
            if (cluster==k[j]) {
                a[i]=k[j];
            }
        }
    }

    if (clock_gettime(CLOCK_REALTIME, &stop)==-1) perror("clock gettime");
    double time=(stop.tv_sec-start.tv_sec)+(double)(stop.tv_nsec-start.tv_nsec)/1e9;
    printf("Execution time: %f sec\n", time);

    if (!(fp=fopen(output_file, "wb"))) {
        printf("can't open output file\n");
        return 1;
    }

    fwrite(a, sizeof(unsigned char), h*w, fp);
    fclose(fp);
    printf("Successfully written to output.raw!\n");

    return 0;
}

void* update_cluster(void *arg) {
    struct thread_data *data=(struct thread_data*)arg;
    int sums[num_clusters]={data->k[0], data->k[1], data->k[2], 
                            data->k[3], data->k[4], data->k[5]};
    int nums[num_clusters]={0, 0, 0, 0, 0, 0};
    for (int i=data->start; i<data->end; i++) {
        int num=data->a[i];
        int cluster=find_cluster(num, data->k);
        for (int j=0; j<num_clusters; j++) {
            if (cluster==data->k[j]) {
                sums[j]+=num;
                nums[j]++;
            }
        }
    }
    pthread_mutex_lock(data->mutex);
    *data->r = *data->r + 1;
    if (*data->r==data->p)
        pthread_cond_broadcast(data->cond);
    else
        pthread_cond_wait(data->cond, data->mutex);
    
    for (int i=0; i<num_clusters; i++) {
        data->sums[i]+=sums[i];
        data->nums[i]+=nums[i];
        data->k[i] = data->sums[i] / data->nums[i];
    }
    pthread_mutex_unlock(data->mutex);
    pthread_exit(NULL);
}

int abs(int x) {
    if (x<0) return -x;
    return x;
}

int find_cluster(int x, int *k) {
    int i, min=99999, cluster;
    for (i=0; i<num_clusters; i++) {
        int diff=abs(x-k[i]);
        if (diff<min) {
            min=diff;
            cluster=k[i];
        }
    }
    return cluster;
}