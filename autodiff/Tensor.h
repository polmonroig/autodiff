//
// Created by pol on 11/8/19.
//

#ifndef AUTODIFF_TENSOR_H
#define AUTODIFF_TENSOR_H

#include <vector>
#include "Variable.h"

namespace autodiff{

    class Variable;
    class Tape;

    class Tensor {
    private:
        std::vector<uint> shape;
        uint size;
        std::vector<autodiff::Variable> parameters;
        autodiff::Tape *tape;
    public:
        explicit Tensor(std::vector<uint> const& shape, Tape* tape);
        explicit Tensor(std::vector<float> const& values);


        bool requires_grad() const;

        void set_var(int i, Variable const &variable);

        autodiff::Variable dot(Tensor const& t2);

        Tensor operator*(const Variable &v2);

        Tensor operator+(const Tensor &t2);

        Tensor operator-(const Tensor &t2);

        autodiff::Variable at(int i);

        void add(int i, Variable const& variable);
    };
}



#endif //AUTODIFF_TENSOR_H
