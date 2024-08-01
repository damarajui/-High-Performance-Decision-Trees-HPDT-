#ifndef PARALLEL_CPU_HPP
#define PARALLEL_CPU_HPP

#include "decision_tree.hpp"
#include <thread>

class ParallelDecisionTree : public DecisionTree {
public:
    ParallelDecisionTree(int num_threads);
    void buildTree(const std::vector<std::vector<std::string>>& data, const std::vector<std::string>& attributes) override;

private:
    int num_threads;
    Node* buildTreeParallel(const std::vector<std::vector<std::string>>& data, const std::vector<std::string>& attributes);
    int findBestAttributeParallel(const std::vector<std::vector<std::string>>& data, const std::vector<std::string>& attributes);
};

#endif // PARALLEL_CPU_HPP