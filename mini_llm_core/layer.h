#pragma once
#include "matrix.h"

namespace llm
{
    // Composite-pattern contract: anything with (or made up of) learnable
    // parameters can take an SGD step and clear its accumulated gradients.
    // Deliberately does NOT try to unify forward()/backward() the way the
    // earlier commented-out sketch did -- those differ in signature across
    // layers (Embedding::forward takes vector<size_t>, not Matrix), so forcing
    // a common signature would mean an awkward wrapper nobody asked for.
    // update()/zero_grad() don't have that problem: every layer, leaf or
    // composite, can implement exactly this.
    class Layer
    {
    public:
        virtual ~Layer() = default;
        virtual void update(scalar lr) = 0;
        virtual void zero_grad() = 0;
    };
}