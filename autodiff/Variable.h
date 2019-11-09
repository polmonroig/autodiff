//
// Created by pol on 11/8/19.
//

#ifndef AUTODIFF_VARIABLE_H
#define AUTODIFF_VARIABLE_H

#include <vector>
#include <string>
#include "Tape.h"



namespace autodiff{

    class Tape;
    class Tensor;

    class Variable {

    private:
        std::string name;
        float value{};
        uint index{};

        autodiff::Tape *tape{};

    public:

        Variable();

        Variable(float value, std::string const& name="var");

        Variable(float value, std::string const &name, autodiff::Tape *tape, uint index);

        autodiff::Tensor grad();

        [[nodiscard]] float get_value() const;

        uint get_index() const;

        bool requires_grad() const;

        /*
         * OPERATORS
         * */

        Variable sin() const;

        Variable pow(float power) const;

        Variable operator*(const Variable &v2);

        Variable operator+(const Variable &v2);

        Variable operator-(const Variable &v2);

    };
}




#endif //AUTODIFF_VARIABLE_H
