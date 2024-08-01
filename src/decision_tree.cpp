#include "decision_tree.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

DecisionTree::DecisionTree() : root(nullptr) {}

DecisionTree::~DecisionTree() {
    deleteTree(root);
}

void DecisionTree::buildTree(const std::vector<std::vector<std::string>>& data, const std::vector<std::string>& attributes) {
    root = buildTreeRecursive(data, attributes);
}

Node* DecisionTree::buildTreeRecursive(const std::vector<std::vector<std::string>>& data, const std::vector<std::string>& attributes) {
    Node* node = new Node();

    // Check if all instances have the same label
    bool same_label = true;
    std::string first_label = data[0].back();
    for (const auto& instance : data) {
        if (instance.back() != first_label) {
            same_label = false;
            break;
        }
    }

    if (same_label) {
        node->is_leaf = true;
        node->label = first_label;
        return node;
    }

    // Check if there are no more attributes to split on
    if (attributes.empty()) {
        node->is_leaf = true;
        // Find the most common label
        std::unordered_map<std::string, int> label_count;
        for (const auto& instance : data) {
            label_count[instance.back()]++;
        }
        node->label = std::max_element(label_count.begin(), label_count.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; })->first;
        return node;
    }

    // Find the best attribute to split on
    int best_attribute_index = findBestAttribute(data, attributes);
    node->attribute = attributes[best_attribute_index];

    // Create subsets based on the best attribute
    std::unordered_map<std::string, std::vector<std::vector<std::string>>> subsets;
    for (const auto& instance : data) {
        subsets[instance[best_attribute_index]].push_back(instance);
    }

    // Create new attributes list without the best attribute
    std::vector<std::string> new_attributes = attributes;
    new_attributes.erase(new_attributes.begin() + best_attribute_index);

    // Recursively build subtrees
    for (const auto& subset : subsets) {
        node->children[subset.first] = buildTreeRecursive(subset.second, new_attributes);
    }

    return node;
}

double DecisionTree::calculateEntropy(const std::vector<std::vector<std::string>>& data) {
    std::unordered_map<std::string, int> label_count;
    for (const auto& instance : data) {
        label_count[instance.back()]++;
    }

    double entropy = 0.0;
    int total_instances = data.size();
    for (const auto& count : label_count) {
        double probability = static_cast<double>(count.second) / total_instances;
        entropy -= probability * std::log2(probability);
    }

    return entropy;
}

double DecisionTree::calculateInformationGain(const std::vector<std::vector<std::string>>& data, int attribute_index) {
    double total_entropy = calculateEntropy(data);
    std::unordered_map<std::string, std::vector<std::vector<std::string>>> subsets;

    for (const auto& instance : data) {
        subsets[instance[attribute_index]].push_back(instance);
    }

    double weighted_entropy = 0.0;
    int total_instances = data.size();
    for (const auto& subset : subsets) {
        double weight = static_cast<double>(subset.second.size()) / total_instances;
        weighted_entropy += weight * calculateEntropy(subset.second);
    }

    return total_entropy - weighted_entropy;
}

int DecisionTree::findBestAttribute(const std::vector<std::vector<std::string>>& data, const std::vector<std::string>& attributes) {
    double max_gain = -1.0;
    int best_attribute_index = -1;

    for (size_t i = 0; i < attributes.size(); ++i) {
        double gain = calculateInformationGain(data, i);
        if (gain > max_gain) {
            max_gain = gain;
            best_attribute_index = i;
        }
    }

    return best_attribute_index;
}

std::string DecisionTree::classify(const std::vector<std::string>& instance) {
    Node* current = root;
    while (!current->is_leaf) {
        auto it = std::find(instance.begin(), instance.end(), current->attribute);
        if (it == instance.end()) {
            // Attribute not found in instance, return the most common label
            return current->label;
        }
        size_t attribute_index = std::distance(instance.begin(), it);
        std::string attribute_value = instance[attribute_index + 1];
        if (current->children.find(attribute_value) == current->children.end()) {
            // If the attribute value is not found in the tree, return the most common label
            return current->label;
        }
        current = current->children[attribute_value];
    }
    return current->label;
}

void DecisionTree::deleteTree(Node* node) {
    if (node == nullptr) return;
    for (auto& child : node->children) {
        deleteTree(child.second);
    }
    delete node;
}