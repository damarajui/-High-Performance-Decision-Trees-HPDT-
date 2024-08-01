#ifndef PARALLEL_GPU_CUH
#define PARALLEL_GPU_CUH

#include "decision_tree.hpp"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

class GPUDecisionTree : public DecisionTree {
public:
    GPUDecisionTree();
    void buildTree(const std::vector<std::vector<std::string>>& data, const std::vector<std::string>& attributes) override;
    std::string classify(const std::vector<std::string>& instance) override;

private:
    Node* buildTreeGPU(thrust::device_vector<int>& data, thrust::device_vector<int>& labels, int num_attributes);
    __device__ double calculateEntropyGPU(const int* labels, int num_instances);
    __device__ double calculateInformationGainGPU(const int* data, const int* labels, int num_instances, int attribute_index);
    std::string classifyGPU(const std::vector<int>& instance);
};

#endif // PARALLEL_GPU_CUH