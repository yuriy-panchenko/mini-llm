#pragma once
#include "tensor.h"

namespace llm
{
	class Linear
	{
		Tensor m_Ws, m_Bias;

	public:
		Tensor forward(const Tensor& input) const;
	};
}