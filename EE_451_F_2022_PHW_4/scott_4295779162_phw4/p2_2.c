#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char** argv)
{
	int size = 4, rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int sum = 0;
	int total_sum = 0;

	if (rank == 0)
	{
		FILE *in_file;
		in_file = fopen("numbers.txt", "r");
		if (in_file == NULL)
			return 1;
		int arr[64];
		for (int i = 0; i < 64; i++)
			fscanf(in_file, "%d", &arr[i]);
		fclose(in_file);

		int ierr;
		ierr = MPI_Send(&arr, 64, MPI_INT, 1, 69, MPI_COMM_WORLD);
		if (ierr != 0) {
			printf("send error\n");
			return 1;
		}
		ierr = MPI_Send(&arr, 64, MPI_INT, 2, 69, MPI_COMM_WORLD);
		if (ierr != 0) {
			printf("send error\n");
			return 1;
		}
		ierr = MPI_Send(&arr, 64, MPI_INT, 3, 69, MPI_COMM_WORLD);
		if (ierr != 0) {
			printf("send error\n");
			return 1;
		}

		for (int i = 0; i < 16; i++)
			sum += arr[i];
	}
	else if (rank == 1)
	{
		int arr[64];
		int ierr = MPI_Recv(&arr, 64, MPI_INT, 0, 69, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		if (ierr != 0) {
			printf("%d recv error\n");
			return 1;
		}

		for (int i = 16; i < 32; i++)
			sum += arr[i];
	}
	else if (rank == 2)
	{
		int arr[64];
		int ierr = MPI_Recv(&arr, 64, MPI_INT, 0, 69, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		if (ierr != 0) {
			printf("%d recv error\n");
			return 1;
		}

		for (int i = 32; i < 48; i++)
			sum += arr[i];
	}
	else // rank == 3
	{
		int arr[64];
		int ierr = MPI_Recv(&arr, 64, MPI_INT, 0, 69, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		if (ierr != 0) {
			printf("%d recv error\n");
			return 1;
		}

		for (int i = 48; i < 64; i++)
			sum += arr[i];
	}

	MPI_Reduce(&sum, &total_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	if (rank == 0)
		printf("Total Sum: %d\n", total_sum);

	MPI_Finalize();
	return 0;
}
















