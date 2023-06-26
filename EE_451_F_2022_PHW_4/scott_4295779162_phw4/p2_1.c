#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char** argv)
{
	int size = 4, rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	FILE *in_file;
	in_file = fopen("numbers.txt", "r");
	if (in_file == NULL)
		return 1;
	int arr[64];
	for (int i = 0; i < 64; i++)
		fscanf(in_file, "%d", &arr[i]);
	fclose(in_file);

	if (rank == 0)
	{
		int sum0 = 0;
		for (int i = 0; i < 16; i++)
			sum0 += arr[i];

		int sum1, sum2, sum3, ierr;
		ierr = MPI_Recv(&sum1, 1, MPI_INT, 1, 69, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		if (ierr != 0) {
			printf("recv error!\n");
			return 1;
		}
		ierr = MPI_Recv(&sum2, 1, MPI_INT, 2, 69, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		if (ierr != 0) {
			printf("recv error!\n");
			return 1;
		}
		ierr = MPI_Recv(&sum3, 1, MPI_INT, 3, 69, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		if (ierr != 0) {
			printf("recv error!\n");
			return 1;
		}

		int total_sum = sum0 + sum1 + sum2 + sum3;
		printf("Total Sum: %d\n", total_sum);
	}
	else if (rank == 1)
	{
		int sum = 0;
		for (int i = 16; i < 32; i++)
			sum += arr[i];

		int ierr = MPI_Send(&sum, 1, MPI_INT, 0, 69, MPI_COMM_WORLD);
		if (ierr != 0) {
			printf("send error from %d\n", rank);
			return 1;
		}
	}
	else if (rank == 2)
	{
		int sum = 0;
		for (int i = 32; i < 48; i++)
			sum += arr[i];

		int ierr = MPI_Send(&sum, 1, MPI_INT, 0, 69, MPI_COMM_WORLD);
			if (ierr != 0) {
			printf("send error from %d\n", rank);
			return 1;
		}
	}
	else // rank == 3
	{
		int sum = 0;
		for (int i = 48; i < 64; i++)
			sum += arr[i];

		int ierr = MPI_Send(&sum, 1, MPI_INT, 0, 69, MPI_COMM_WORLD);
		if (ierr != 0) {
			printf("send error from %d\n", rank);
			return 1;
		}
	}

	MPI_Finalize();
	return 0;
}



