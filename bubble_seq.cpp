#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <locale>
#include <omp.h>

void bubbleSortSeq(std::vector<int>& arr, int left, int right) {
    int n = right - left + 1;
    for (int i = 0; i < n - 1; ++i) {
        for (int j = left; j < right - i; ++j) {
            if (arr[j] > arr[j + 1]) {
                std::swap(arr[j], arr[j + 1]);
            }
        }
    }
}

int main() {
    setlocale(LC_ALL, "rus");
    const int length = 1000;
    std::vector<int> arr(length);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(1, 100);
    for (int i = 0; i < length; ++i) {
        arr[i] = dist(gen);
    }
    long long overallSum = 0;
    double overallTime = 0;

    auto start = std::chrono::high_resolution_clock::now();
    for (int j = 0; j < 100; ++j) {
        bubbleSortSeq(arr, 0, length - 1);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Sequential " << std::endl;
    std::cout << "Sorting finished" << std::endl;
    std::cout << "Time taken: " << elapsed.count() << " sec" << std::endl;

    return 0;
}
