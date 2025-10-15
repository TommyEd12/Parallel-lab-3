#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <locale>
#include <mpi.h>

int main(int argc, char** argv) {
    setlocale(LC_ALL, "rus");
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const int length = 100000;
    std::vector<int> arr, secondArr, resArr;
    std::vector<int> local_arr, local_secondArr, local_resArr;


    if (world_rank == 0) {
        arr.resize(length);
        secondArr.resize(length);
        resArr.resize(length);

        std::mt19937 gen(42);
        std::uniform_int_distribution<int> dist(1, 1000);
        std::uniform_int_distribution<int> secondDist(1, 200);

        for (int i = 0; i < length; ++i) {
            arr[i] = dist(gen);
        }
        for (int i = 0; i < length; ++i) {
            secondArr[i] = secondDist(gen);
        }
    }


    int local_length = length / world_size;
    int remainder = length % world_size;


    if (world_rank == world_size - 1) {
        local_length += remainder;
    }

    local_arr.resize(local_length);
    local_secondArr.resize(local_length);
    local_resArr.resize(local_length);


    std::vector<int> sendcounts(world_size);
    std::vector<int> displs(world_size);

    if (world_rank == 0) {
        int offset = 0;
        for (int i = 0; i < world_size; ++i) {
            sendcounts[i] = length / world_size;
            if (i == world_size - 1) {
                sendcounts[i] += length % world_size;
            }
            displs[i] = offset;
            offset += sendcounts[i];
        }
    }

    double overallTime = 0;
    long long global_sum = 0;


    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int j = 0; j < 100; ++j) {

        MPI_Scatterv(arr.data(), sendcounts.data(), displs.data(), MPI_INT,
            local_arr.data(), local_length, MPI_INT,
            0, MPI_COMM_WORLD);

        MPI_Scatterv(secondArr.data(), sendcounts.data(), displs.data(), MPI_INT,
            local_secondArr.data(), local_length, MPI_INT,
            0, MPI_COMM_WORLD);


        long long local_sum = 0;
        for (int i = 0; i < local_length; ++i) {
            local_resArr[i] = local_arr[i] + local_secondArr[i];
            local_resArr[i] = local_arr[i] - local_secondArr[i];
            local_resArr[i] = local_arr[i] * local_secondArr[i];
            if (local_secondArr[i] != 0) {
                local_resArr[i] = local_arr[i] / local_secondArr[i];
            }
            local_sum += local_resArr[i];
        }


        MPI_Gatherv(local_resArr.data(), local_length, MPI_INT,
            resArr.data(), sendcounts.data(), displs.data(), MPI_INT,
            0, MPI_COMM_WORLD);


        MPI_Reduce(&local_sum, &global_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;


    double local_time = elapsed.count();
    double max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);


    if (world_rank == 0) {
        std::cout << "Processes " << world_size << std::endl;
        std::cout << "Sum elements: " << global_sum << std::endl;
        std::cout << "Time taken 100 launches: " << max_time << " sec" << std::endl;


        std::cout << "First five elements: ";
        for (int i = 0; i < 5 && i < length; ++i) {
            std::cout << resArr[i] << " ";
        }
        std::cout << std::endl;
    }

    MPI_Finalize();
    return 0;
}
