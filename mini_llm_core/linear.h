#pragma once
#include "tensor.h"

namespace llm
{
	class Linear
	{
		Tensor m_Ws, m_Bias;

	public:
		Linear(Tensor const& w, Tensor const& bias);

		Tensor forward(const Tensor& input) const;
	};
}