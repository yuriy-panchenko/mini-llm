#pragma once
#include "tensor.h"

namespace llm
{
    class RMSNorm
    {
        Tensor m_Gamma;
        scalar m_Eps;

    public:
        RMSNorm(Tensor const& gamma, scalar eps = 1e-5f);
        Tensor forward(Tensor const& input) const;
    };
}