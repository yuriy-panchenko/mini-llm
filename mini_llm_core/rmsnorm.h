#pragma once
#include "tensor.h"
#include "vector.h"

namespace llm
{
    class RMSNorm
    {
        Vector m_Gamma;
        scalar m_Eps;

    public:
        RMSNorm(Vector const& gamma, scalar eps = 1e-5f);
        Matrix forward(Matrix const& input) const;
    };
}