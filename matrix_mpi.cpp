#include <iostream>
#include <random>
#include <chrono>
#include <cstdint>
#include <mpi.h>

int main(int argc, char** argv) {
    setlocale(LC_ALL, "rus");
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const int length = 1000;
    const int runs = 100;


    int rows_per_process = length / world_size;
    int remainder = length % world_size;


    int start_row = world_rank * rows_per_process;
    int end_row = start_row + rows_per_process;


    if (world_rank == world_size - 1) {
        end_row += remainder;
    }

    int local_rows = end_row - start_row;


    int** local_arr = new int* [local_rows];
    int* local_arr_block = new int[static_cast<size_t>(local_rows) * length];
    for (int i = 0; i < local_rows; ++i) {
        local_arr[i] = local_arr_block + static_cast<size_t>(i) * length;
    }

    int** local_secondArr = new int* [local_rows];
    int* local_second_block = new int[static_cast<size_t>(local_rows) * length];
    for (int i = 0; i < local_rows; ++i) {
        local_secondArr[i] = local_second_block + static_cast<size_t>(i) * length;
    }

    int** local_resArr = new int* [local_rows];
    int* local_res_block = new int[static_cast<size_t>(local_rows) * length];
    for (int i = 0; i < local_rows; ++i) {
        local_resArr[i] = local_res_block + static_cast<size_t>(i) * length;
    }

    std::mt19937 rng(42 + world_rank); 
    std::uniform_int_distribution<int> d1(1, 100);
    std::uniform_int_distribution<int> d2(1, 200);

    for (int i = 0; i < local_rows; ++i) {
        for (int j = 0; j < length; ++j) {
            local_arr[i][j] = d1(rng);
            local_secondArr[i][j] = d2(rng);
        }
    }

    long long overallSum = 0;
    double local_time = 0.0;


    MPI_Barrier(MPI_COMM_WORLD);
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int r = 0; r < runs; ++r) {
        long long local_sum = 0;

        for (int i = 0; i < local_rows; ++i) {
            for (int j = 0; j < length; ++j) {
                int a = local_arr[i][j];
                int b = local_secondArr[i][j];

                int s = a + b;
                int d = a - b;
                long long m = 1LL * a * b;
                int div = (b != 0) ? (a / b) : 0;

                local_resArr[i][j] = s;

                local_sum += s;
                local_sum += d;
                local_sum += m;
                local_sum += div;
            }
        }

        overallSum += local_sum;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t1 - t0;
    local_time = elapsed.count();


    long long global_sum = 0;
    double max_time = 0.0;

    MPI_Reduce(&overallSum, &global_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);


    if (world_rank == 0) {
        std::cout << "Processes: " << world_size << std::endl;
        std::cout << "Sum elements: " << global_sum << std::endl;
        std::cout << "Time taken for " << runs << " launches: " << max_time << " sec" << std::endl;
 
        std::cout << std::endl;
    }

    delete[] local_arr_block;
    delete[] local_arr;
    delete[] local_second_block;
    delete[] local_secondArr;
    delete[] local_res_block;
    delete[] local_resArr;

    MPI_Finalize();
    return 0;
}
