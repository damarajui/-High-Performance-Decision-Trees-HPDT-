#include "parallel_gpu.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <unordered_map>

__global__ void calculateEntropyKernel(const int* labels, int num_instances, double* entropy) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_instances) {
        int label = labels[tid];
        atomicAdd(&entropy[label], 1.0 / num_instances);
    }
}

__global__ void calculateInformationGainKernel(const int* data, const int* labels, int num_instances, int num_attributes, double* gains) {
    int attribute_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (attribute_index < num_attributes) {
        gains[attribute_index] = calculateInformationGainGPU(data, labels, num_instances, attribute_index);
    }
}

__global__ void checkBaseCasesKernel(const int* labels, int num_instances, bool* is_leaf, int* majority_label) {
    __shared__ int local_counts[256];
    int tid = threadIdx.x;
    local_counts[tid] = 0;

    for (int i = tid; i < num_instances; i += blockDim.x) {
        atomicAdd(&local_counts[labels[i]], 1);
    }

    __syncthreads();

    if (tid == 0) {
        int max_count = 0;
        int max_label = 0;
        bool all_same = true;
        for (int i = 0; i < 256; ++i) {
            if (local_counts[i] > 0) {
                if (local_counts[i] > max_count) {
                    max_count = local_counts[i];
                    max_label = i;
                }
                if (local_counts[i] != num_instances) {
                    all_same = false;
                }
            }
        }
        *is_leaf = all_same || (num_instances == 0);
        *majority_label = max_label;
    }
}

GPUDecisionTree::GPUDecisionTree() : DecisionTree() {}

void GPUDecisionTree::buildTree(const std::vector<std::vector<std::string>>& data, const std::vector<std::string>& attributes) {
    // Convert string data to integer data for GPU processing
    std::unordered_map<std::string, int> value_to_int;
    int next_id = 0;
    std::vector<std::vector<int>> int_data(data.size(), std::vector<int>(data[0].size()));
    
    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < data[i].size(); ++j) {
            if (value_to_int.find(data[i][j]) == value_to_int.end()) {
                value_to_int[data[i][j]] = next_id++;
            }
            int_data[i][j] = value_to_int[data[i][j]];
        }
    }

    // Prepare data for GPU
    thrust::host_vector<int> h_data;
    thrust::host_vector<int> h_labels;
    for (const auto& instance : int_data) {
        h_data.insert(h_data.end(), instance.begin(), instance.end() - 1);
        h_labels.push_back(instance.back());
    }

    thrust::device_vector<int> d_data = h_data;
    thrust::device_vector<int> d_labels = h_labels;

    // Build the tree using GPU
    root = buildTreeGPU(d_data, d_labels, attributes.size() - 1);
}

Node* GPUDecisionTree::buildTreeGPU(thrust::device_vector<int>& data, thrust::device_vector<int>& labels, int num_attributes) {
    Node* node = new Node();

    // Check for base cases
    thrust::host_vector<int> h_labels = labels;
    if (thrust::equal(h_labels.begin() + 1, h_labels.end(), h_labels.begin()) || num_attributes == 0) {
        node->is_leaf = true;
        node->label = std::to_string(thrust::reduce(h_labels.begin(), h_labels.end(), 0, thrust::maximum<int>()));
        return node;
    }

    // Find best attribute to split on
    thrust::device_vector<double> d_gains(num_attributes);
    int block_size = 256;
    int num_blocks = (num_attributes + block_size - 1) / block_size;

    calculateInformationGainKernel<<<num_blocks, block_size>>>(
        thrust::raw_pointer_cast(data.data()),
        thrust::raw_pointer_cast(labels.data()),
        labels.size(),
        num_attributes,
        thrust::raw_pointer_cast(d_gains.data())
    );

    thrust::host_vector<double> h_gains = d_gains;
    int best_attribute = thrust::max_element(h_gains.begin(), h_gains.end()) - h_gains.begin();

    node->attribute = std::to_string(best_attribute);

    // Create subsets
    thrust::device_vector<int> d_subset_indices(data.size() / num_attributes);
    thrust::device_vector<int> d_subset_counts(num_attributes);

    auto subset_end = thrust::reduce_by_key(
        thrust::device,
        thrust::make_transform_iterator(thrust::counting_iterator<int>(0), [=] __device__ (int i) { return data[i * num_attributes + best_attribute]; }),
        thrust::make_transform_iterator(thrust::counting_iterator<int>(0), [=] __device__ (int i) { return data[i * num_attributes + best_attribute]; }) + data.size() / num_attributes,
        thrust::make_transform_iterator(thrust::counting_iterator<int>(0), [=] __device__ (int i) { return i; }),
        d_subset_indices.begin(),
        d_subset_counts.begin()
    );

    int num_subsets = subset_end.first - d_subset_indices.begin();

    // Create subsets using the d_subset_indices and d_subset_counts
    for (const auto& subset : subset_indices) {
        thrust::device_vector<int> d_subset_data(subset.second.size() * (num_attributes - 1));
        thrust::device_vector<int> d_subset_labels(subset.second.size());

        // Copy subset data and labels to device vectors
        for (size_t i = 0; i < subset.second.size(); ++i) {
            int idx = subset.second[i];
            for (int j = 0, k = 0; j < num_attributes; ++j) {
                if (j != best_attribute) {
                    d_subset_data[i * (num_attributes - 1) + k] = h_data[idx * num_attributes + j];
                    ++k;
                }
            }
            d_subset_labels[i] = h_labels[idx];
        }

        node->children[std::to_string(subset.first)] = buildTreeGPU(d_subset_data, d_subset_labels, num_attributes - 1);
    }

    return node;
}

__device__ double GPUDecisionTree::calculateEntropyGPU(const int* labels, int num_instances) {
    __shared__ double local_entropy[256];
    int tid = threadIdx.x;
    local_entropy[tid] = 0.0;

    for (int i = tid; i < num_instances; i += blockDim.x) {
        int label = labels[i];
        double prob = 1.0 / num_instances;
        atomicAdd(&local_entropy[label], prob * __log2f(prob));
    }

    __syncthreads();

    if (tid == 0) {
        double entropy = 0.0;
        for (int i = 0; i < 256; ++i) {
            entropy -= local_entropy[i];
        }
        return entropy;
    }
    return 0.0;
}

__device__ double GPUDecisionTree::calculateInformationGainGPU(const int* data, const int* labels, int num_instances, int attribute_index) {
    __shared__ double local_gain[256];
    int tid = threadIdx.x;
    local_gain[tid] = 0.0;

    double total_entropy = calculateEntropyGPU(labels, num_instances);

    for (int i = tid; i < num_instances; i += blockDim.x) {
        int attribute_value = data[i * gridDim.x + attribute_index];
        double subset_entropy = calculateEntropyGPU(&labels[i], 1);
        atomicAdd(&local_gain[attribute_value], subset_entropy / num_instances);
    }

    __syncthreads();

    if (tid == 0) {
        double weighted_entropy = 0.0;
        for (int i = 0; i < 256; ++i) {
            weighted_entropy += local_gain[i];
        }
        return total_entropy - weighted_entropy;
    }
    return 0.0;
}