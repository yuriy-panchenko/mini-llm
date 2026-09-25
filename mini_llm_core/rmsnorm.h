#pragma once
#include "vector.h"
#include "matrix.h"

namespace llm
{
    class RMSNorm
    {
        Vector m_Gamma,m_dGamma;
        scalar m_Eps;
        Matrix m_LastInput;

    public:
        RMSNorm(Vector const& gamma, scalar eps = 1e-5f);
        Matrix forward(Matrix const& input);
        Matrix backward(Matrix const& dOut);
    };
}