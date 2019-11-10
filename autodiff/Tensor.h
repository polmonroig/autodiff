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
        std::vector<autodiff::Tensor> sub_tensors;
        uint is_leaf;
        bool _requires_grad;

    public:
        explicit Tensor(std::vector<uint> const& shape, bool requires_grad = false);

        static autodiff::Tensor rand(std::vector<uint> const& shape, bool requires_grad);

        void record_parameters();


        void set_var(std::vector<uint> const& indexes, autodiff::Variable const& var);

        void add(std::vector<uint> const &indexes, autodiff::Variable const& var);

        autodiff::Variable at(std::vector<uint> const &indexes);

        void fill_random(bool requires_grad);

        Variable dot(const Tensor &t2);

        bool requires_grad() const;

        Tensor matmul( Tensor &t2);
    };

    std::ostream& operator<<(std::ostream&, const autodiff::Tensor&);
}



#endif //AUTODIFF_TENSOR_H
