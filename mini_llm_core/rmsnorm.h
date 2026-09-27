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

        // Adam moment estimates -- persist across steps, untouched by zero_grad()
        Vector m_mGamma, m_vGamma;
        size_t m_T{ 0 };

    public:
        RMSNorm(Vector const& gamma, scalar eps = 1e-5f);
        Matrix forward(Matrix const& input);
        Matrix backward(Matrix const& dOut);

        void update(scalar lr) override;
        void zero_grad() override;
    };
}