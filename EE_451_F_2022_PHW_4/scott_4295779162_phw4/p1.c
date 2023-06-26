#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char** argv) {
    int size = 4;
	int rank = 0;
    MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	int msg = 0;
	if (rank == 0) {
		msg = 451;
//		printf("Process %d: Initially Msg = %d\n", rank, msg);
		MPI_Send(&msg, 1, MPI_INT, 1, 69, MPI_COMM_WORLD);
	} else if (rank == 1) {
		MPI_Recv(&msg, 1, MPI_INT, 0, 69, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		int *msg_ptr = &msg;
		*msg_ptr = *msg_ptr + 1;
//		printf("Process %d: Msg = %d\n", rank, msg);
		MPI_Send(&msg, 1, MPI_INT, 2, 69, MPI_COMM_WORLD);
	} else if (rank == 2) {
		MPI_Recv(&msg, 1, MPI_INT, 1, 69, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		int *msg_ptr = &msg;
		*msg_ptr = *msg_ptr + 1;
//		printf("Process %d: Msg = %d\n", rank, msg);
		MPI_Send(&msg, 1, MPI_INT, 3, 69, MPI_COMM_WORLD);
	} else { // rank == 3
		MPI_Recv(&msg, 1, MPI_INT, 2, 69, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		int *msg_ptr = &msg;
		*msg_ptr = *msg_ptr + 1;
//		printf("Process %d: Msg = %d\n", rank, msg);
		MPI_Send(&msg, 1, MPI_INT, 0, 69, MPI_COMM_WORLD);
	}

	if (rank == 0) {
		MPI_Recv(&msg, 1, MPI_INT, 3, 69, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		printf("Process %d: Received Msg = %d. Done!\n", rank, msg);
	}
	
	MPI_Finalize();
	return 0;
}
