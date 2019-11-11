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
        float value = 0.0;
        uint index{};
        bool _requires_grad{};


    public:

        Variable();

        Variable(float value, bool requires_grad=false, std::string const& name="var");

        autodiff::Tensor grad();

        [[nodiscard]] float get_value() const;

        uint get_index() const;

        bool requires_grad() const;

        void record_var(int i);

        /*
         * OPERATORS
         * */

        Variable sin() const;

        Variable pow(float power) const;

        Variable operator*(const Variable &v2)const;

        Variable operator+(const Variable &v2) const;

        Variable operator-(const Variable &v2) const;
        Variable operator/(const Variable &v2)const;

        static Variable sigmoid(const Variable &var);

        Variable abs() const;
    };
}




#endif //AUTODIFF_VARIABLE_H
