#include "decision_tree.hpp"
#include "parallel_cpu.hpp"
#include "parallel_gpu.cuh"
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>

std::vector<std::vector<std::string>> readCSV(const std::string& filename) {
    std::vector<std::vector<std::string>> data;
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line)) {
        std::vector<std::string> row;
        std::stringstream ss(line);
        std::string cell;

        while (std::getline(ss, cell, ',')) {
            row.push_back(cell);
        }
        data.push_back(row);
    }

    return data;
}

int main() {
    // Read the dataset
    std::string filename = "data/your_dataset.csv";
    auto data = readCSV(filename);

    // Extract attributes (assuming the first row contains attribute names)
    std::vector<std::string> attributes(data[0].begin(), data[0].end() - 1);
    data.erase(data.begin());

    // Serial implementation
    {
        DecisionTree serial_tree;
        auto start = std::chrono::high_resolution_clock::now();
        serial_tree.buildTree(data, attributes);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Serial implementation time: " << elapsed.count() << " seconds" << std::endl;
    }

    // Parallel CPU implementation
    {
        ParallelDecisionTree parallel_cpu_tree(4); // Use 4 threads
        auto start = std::chrono::high_resolution_clock::now();
        parallel_cpu_tree.buildTree(data, attributes);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Parallel CPU implementation time: " << elapsed.count() << " seconds" << std::endl;
    }

    // GPU implementation
    {
        GPUDecisionTree gpu_tree;
        auto start = std::chrono::high_resolution_clock::now();
        gpu_tree.buildTree(data, attributes);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "GPU implementation time: " << elapsed.count() << " seconds" << std::endl;
    }

    // Test classification
    std::vector<std::string> test_instance = {"sunny", "hot", "high", "weak"};
    std::cout << "Prediction for test instance:" << std::endl;
    std::cout << "Serial: " << serial_tree.classify(test_instance) << std::endl;
    std::cout << "Parallel CPU: " << parallel_cpu_tree.classify(test_instance) << std::endl;
    std::cout << "GPU: " << gpu_tree.classify(test_instance) << std::endl;

    return 0;
}