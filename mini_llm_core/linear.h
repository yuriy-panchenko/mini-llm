#pragma once
#include "matrix.h"
#include "vector.h"
//#include "tensor.h"

namespace llm
{
	class Linear
	{
		Matrix m_Ws, m_dWs, m_LastInput;
		Vector m_Bias, m_dBias;

	public:
		Linear(Matrix const& w, Vector const& bias);

		Matrix forward(const Matrix& input);
		Matrix backward(Matrix const& dOut);
	};
}