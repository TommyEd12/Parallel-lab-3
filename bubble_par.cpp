#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <locale>
#include <mpi.h>
#include <algorithm>

void bubbleSortPar(std::vector<int>& arr, int left, int right) {
    int n = right - left + 1;
    for (int i = 0; i < n - 1; ++i) {
        for (int j = left; j < right - i; ++j) {
            if (arr[j] > arr[j + 1]) {
                std::swap(arr[j], arr[j + 1]);
            }
        }
    }
}


void merge(std::vector<int>& arr, int left, int mid, int right) {
    std::vector<int> temp(right - left + 1);
    int i = left, j = mid + 1, k = 0;

    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        }
        else {
            temp[k++] = arr[j++];
        }
    }

    while (i <= mid) {
        temp[k++] = arr[i++];
    }

    while (j <= right) {
        temp[k++] = arr[j++];
    }

    for (int idx = 0; idx < k; ++idx) {
        arr[left + idx] = temp[idx];
    }
}

int main(int argc, char** argv) {
    setlocale(LC_ALL, "rus");
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const int length = 1000;
    std::vector<int> arr;
    std::vector<int> local_arr;


    if (world_rank == 0) {
        arr.resize(length);
        std::mt19937 gen(42);
        std::uniform_int_distribution<int> dist(1, 100);
        for (int i = 0; i < length; ++i) {
            arr[i] = dist(gen);
        }
   
    }

    double overallTime = 0;

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int j = 0; j < 100; ++j) {
        if (world_rank == 0) {
            std::mt19937 gen(42 + j);
            std::uniform_int_distribution<int> dist(1, 100);
            for (int i = 0; i < length; ++i) {
                arr[i] = dist(gen);
            }
        }


        int local_size = length / world_size;
        local_arr.resize(local_size);


        MPI_Scatter(arr.data(), local_size, MPI_INT,
            local_arr.data(), local_size, MPI_INT,
            0, MPI_COMM_WORLD);


        bubbleSortPar(local_arr, 0, local_size - 1);


        MPI_Gather(local_arr.data(), local_size, MPI_INT,
            arr.data(), local_size, MPI_INT,
            0, MPI_COMM_WORLD);


        if (world_rank == 0) {

            int remaining = length % world_size;
            if (remaining > 0) {
                bubbleSortPar(arr, length - remaining, length - 1);
            }

            for (int step = 1; step < world_size; step *= 2) {
                for (int i = 0; i + step < world_size; i += 2 * step) {
                    int left = i * local_size;
                    int mid = (i + step) * local_size - 1;
                    int right = std::min((i + 2 * step) * local_size - 1, length - 1);
                    merge(arr, left, mid, right);
                }
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;


    double local_time = elapsed.count();
    double max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        std::cout << "Number of processes: " << world_size << std::endl;
        std::cout << "Sorting finished" << std::endl;
        std::cout << "Time taken for 100 iterations: " << max_time << " sec" << std::endl;

        bool sorted = true;
        for (int i = 1; i < length; ++i) {
            if (arr[i] < arr[i - 1]) {
                sorted = false;
                break;
            }
        }


        std::cout << "First 10 elements: ";
        for (int i = 0; i < 10 && i < length; ++i) {
            std::cout << arr[i] << " ";
        }
        std::cout << std::endl;
    }

    MPI_Finalize();
    return 0;
}
