#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <mpi.h>
#include <locale>

int main(int argc, char** argv) {
    setlocale(LC_ALL, "rus");

    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const int length = 100000;
    std::vector<int> arr;

  
    if (world_rank == 0) {
        arr.resize(length);
        std::mt19937 gen(42);
        std::uniform_int_distribution<int> dist(1, 1000);
        for (int i = 0; i < length; ++i) {
            arr[i] = dist(gen);
        }
    }


    if (world_rank == 0) {
        for (int i = 1; i < world_size; ++i) {
            MPI_Send(arr.data(), length, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
    }
    else {
        arr.resize(length);
        MPI_Recv(arr.data(), length, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    long long overallSum = 0;
    double overallTime = 0;

    for (int j = 0; j < 10000; ++j) {
      
        MPI_Barrier(MPI_COMM_WORLD);
        auto start = std::chrono::high_resolution_clock::now();

      
        int chunk_size = length / world_size;
        int start_index = world_rank * chunk_size;
        int end_index = (world_rank == world_size - 1) ? length : start_index + chunk_size;

        long long local_sum = 0;
        for (int i = start_index; i < end_index; ++i) {
            local_sum += arr[i];
        }

       
        long long global_sum = 0;
        MPI_Reduce(&local_sum, &global_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        if (world_rank == 0) {
            overallSum = global_sum;
            overallTime += elapsed.count();
        }
    }

   
    if (world_rank == 0) {
        std::cout << "Processes: " << world_size << std::endl;
        std::cout << "Sum eleemnts " << overallSum << std::endl;
        std::cout << "Time for 10000 launches: " << overallTime << " sec" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
