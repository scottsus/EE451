CC = gcc

all: p1_serial p1a p1b p2_serial p2_parallel p2_random

p1_serial: p1_serial.c
	$(CC) -o p1_serial p1_serial.c

p1a: p1a.c
	$(CC) -lrt -fopenmp -o p1a p1a.c

p1b: p1b.c
	$(CC) -lrt -fopenmp -o p1b p1b.c

run_p1:
	./p1_serial && ./p1a && ./p1b

p2_serial: p2_serial.c
	$(CC) -lrt -fopenmp -o p2_serial p2_serial.c

p2_parallel: p2_parallel.c
	$(CC) -lrt -fopenmp -o p2_parallel p2_parallel.c

p2_random: p2_random.c
	$(CC) -lrt -fopenmp -o p2_random p2_random.c

run_p2:
	./p2_serial && ./p2_parallel && ./p2_random

p3: p3.c
	$(CC) -lrt -fopenmp -o p3 p3.c

run_p3:
	./p3

clean:
	rm p1_serial p1a p1b p2_serial p2_parallel p2_random p3
	