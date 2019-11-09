//
// Created by pol on 11/8/19.
//

#ifndef AUTODIFF_TAPE_H
#define AUTODIFF_TAPE_H

#include <utility>
#include <vector>
#include "Variable.h"
#include "Tensor.h"


namespace autodiff{


    class Variable;
    class Tensor;

    struct Node{
        std::pair<float, float> weights;
        std::pair<uint, uint> children;

        Node(std::pair<float, float> w, std::pair<uint, uint> d) : weights(w), children(d){};
    };

    class Tape {
    private:
        std::vector<Node> tape;

    public:
        autodiff::Variable variable(float value, std::string const& name);

        Node get_node(int i) const;

        uint size() const;

        uint push_leaf();

        uint push_unary(uint child, float weight);

        uint push_binary(Node var);

        autodiff::Tensor tensor1d(std::vector<float> values, const std::string &name);

        void clean();
    };
}



#endif //AUTODIFF_TAPE_H
