#pragma once
#include "matrix.h"
#include "vector.h"
//#include "tensor.h"

namespace llm
{
	class Linear
	{
		Matrix m_Ws;
		Vector m_Bias;

	public:
		Linear(Matrix const& w, Vector const& bias);

		Matrix forward(const Matrix& input) const;
	};
}