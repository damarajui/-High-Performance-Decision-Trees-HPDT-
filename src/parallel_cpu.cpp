#include "parallel_cpu.hpp"
#include <algorithm>
#include <future>

ParallelDecisionTree::ParallelDecisionTree(int num_threads) : num_threads(num_threads) {}

void ParallelDecisionTree::buildTree(const std::vector<std::vector<std::string>>& data, const std::vector<std::string>& attributes) {
    root = buildTreeParallel(data, attributes);
}

Node* ParallelDecisionTree::buildTreeParallel(const std::vector<std::vector<std::string>>& data, const std::vector<std::string>& attributes) {
    Node* node = new Node();

    // Base cases (same as serial version)
    if (allSameClass(data) || attributes.empty()) {
        node->is_leaf = true;
        node->label = getMajorityClass(data);
        return node;
    }

    // Find the best attribute to split on (parallel implementation)
    int best_attribute_index = findBestAttributeParallel(data, attributes);
    node->attribute = attributes[best_attribute_index];

    // Create subsets (same as serial version)
    std::unordered_map<std::string, std::vector<std::vector<std::string>>> subsets;
    for (const auto& instance : data) {
        subsets[instance[best_attribute_index]].push_back(instance);
    }

    // Create new attributes list without the best attribute
    std::vector<std::string> new_attributes = attributes;
    new_attributes.erase(new_attributes.begin() + best_attribute_index);

    // Recursively build subtrees (parallel implementation)
    std::vector<std::future<Node*>> futures;
    for (const auto& subset : subsets) {
        futures.push_back(std::async(std::launch::async, [this, &subset, &new_attributes]() {
            return buildTreeParallel(subset.second, new_attributes);
        }));
    }

    for (size_t i = 0; i < futures.size(); ++i) {
        node->children[subsets.begin()->first] = futures[i].get();
        ++subsets.begin();
    }

    return node;
}

int ParallelDecisionTree::findBestAttributeParallel(const std::vector<std::vector<std::string>>& data, const std::vector<std::string>& attributes) {
    std::vector<double> gains(attributes.size());
    std::vector<std::future<void>> futures;

    for (size_t i = 0; i < attributes.size(); i += num_threads) {
        for (int t = 0; t < num_threads && i + t < attributes.size(); ++t) {
            futures.push_back(std::async(std::launch::async, [this, &data, &gains, i, t]() {
                gains[i + t] = calculateInformationGain(data, i + t);
            }));
        }

        for (auto& future : futures) {
            future.wait();
        }
        futures.clear();
    }

    return std::distance(gains.begin(), std::max_element(gains.begin(), gains.end()));
}