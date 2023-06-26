#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char** argv)
{
	int size = 4, rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	int full_arr[64], arr[16], sums[4], sum = 0;
	if (rank == 0)
	{
		FILE *in_file;
		in_file = fopen("numbers.txt", "r");
		if (in_file == NULL)
			return 1;
		for (int i = 0; i < 64; i++)
			fscanf(in_file, "%d", &full_arr[i]);
		fclose(in_file);
	}

	int err;
	err = MPI_Scatter(full_arr, 16, MPI_INT, arr, 16, MPI_INT, 0, MPI_COMM_WORLD);
	if (err != 0) {
		printf("Scatter failed\n");
		return 1;
	}

	for (int i = 0; i < 16; i++) 
		sum += arr[i];

	err = MPI_Gather(&sum, 1, MPI_INT, sums, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if (err != 0) {
		printf("Gather failed\n");
		return 1;
	}

	if (rank == 0) {
		int total_sum = 0;
		for (int i = 0; i < 4; i++)
			total_sum += sums[i];
		printf("Total Sum: %d\n", total_sum);
	}
	
	MPI_Finalize();
	return 0;
}
