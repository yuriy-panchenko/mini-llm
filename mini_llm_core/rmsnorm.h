#pragma once
#include "vector.h"
#include "layer.h"

namespace llm
{
    class RMSNorm
        :public Layer
    {
        Vector m_Gamma, m_dGamma;
        scalar m_Eps;
        Matrix m_LastInput;

    public:
        RMSNorm(Vector const& gamma, scalar eps = 1e-5f);
        Matrix forward(Matrix const& input);
        Matrix backward(Matrix const& dOut);

        // Inherited via Layer
        void update(scalar lr) override;
        void zero_grad() override;
    };
}