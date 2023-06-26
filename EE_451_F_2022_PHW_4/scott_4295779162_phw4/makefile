CC = mpicc

all: p1 p2_1 p2_2 p2_3

p1: p1.c
	$(CC) p1.c -o p1 && sbatch job0.sl

p2_1: p2_1.c
	$(CC) p2_1.c -o p2_1 && sbatch job1.sl

p2_2: p2_2.c
	$(CC) p2_2.c -o p2_2 && sbatch job2.sl

p2_3: p2_3.c
	$(CC) p2_3.c -o p2_3 && sbatch job3.sl

clean:
	rm p1 p2_1 p2_2 p2_3 mpijob.out mpijob.err
