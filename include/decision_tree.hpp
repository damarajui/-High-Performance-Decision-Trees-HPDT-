#ifndef DECISION_TREE_HPP
#define DECISION_TREE_HPP

#include <vector>
#include <string>
#include <unordered_map>

struct Node {
    std::string attribute;
    std::unordered_map<std::string, Node*> children;
    std::string label;
    bool is_leaf;
};

class DecisionTree {
public:
    DecisionTree();
    ~DecisionTree();
    void buildTree(const std::vector<std::vector<std::string>>& data, const std::vector<std::string>& attributes);
    std::string classify(const std::vector<std::string>& instance);

private:
    Node* root;
    Node* buildTreeRecursive(const std::vector<std::vector<std::string>>& data, const std::vector<std::string>& attributes);
    double calculateEntropy(const std::vector<std::vector<std::string>>& data);
    double calculateInformationGain(const std::vector<std::vector<std::string>>& data, int attribute_index);
    int findBestAttribute(const std::vector<std::vector<std::string>>& data, const std::vector<std::string>& attributes);
    void deleteTree(Node* node);
};

#endif // DECISION_TREE_HPP
