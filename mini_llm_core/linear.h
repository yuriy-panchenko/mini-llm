#pragma once
#include "vector.h"
#include "layer.h"

namespace llm
{
	class Linear
		:public Layer
	{
		Matrix m_Ws, m_dWs, m_LastInput;
		Vector m_Bias, m_dBias;

		// Adam moment estimates -- persist across steps, untouched by zero_grad()
		Matrix m_mWs, m_vWs;
		Vector m_mBias, m_vBias;
		size_t m_T{ 0 };

	public:
		Linear(Matrix const& w, Vector const& bias);

		Matrix forward(const Matrix& input);
		Matrix backward(Matrix const& dOut);

		void update(scalar lr) override;
		void zero_grad() override;
	};
}