#include <iostream>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    double Sum = 0;
    int Size = 0;
    int Rank = 0;
    double PartSum = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &Size);
    MPI_Comm_rank(MPI_COMM_WORLD, &Rank);
    MPI_Status Stat = {};

    int N = atoi(argv[1]);
    int Hop = N / Size;
    int Left = Hop * Rank + 1;
    int Right = Left + Hop;

    for (int i = Left; i < Right; ++i)
    {
        Sum += 1.0 / i;
    }

    if (Rank == 0)
    {
        for (int i = 1; i < Size; ++i)
        {
            MPI_Recv(&PartSum, 1, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &Stat);
            Sum += PartSum;
        }
        
        std::cout << Sum;
    }
    else
    {
        MPI_Send(&Sum, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}