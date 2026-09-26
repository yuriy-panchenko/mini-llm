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

	public:
		Linear(Matrix const& w, Vector const& bias);

		Matrix forward(const Matrix& input);
		Matrix backward(Matrix const& dOut);

		// Inherited via Layer
		void update(scalar lr) override;
		void zero_grad() override;
	};
}