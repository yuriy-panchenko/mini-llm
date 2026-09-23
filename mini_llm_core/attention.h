#pragma once
#include "linear.h"

namespace llm
{
	class Attention
	{
		Linear m_Wq, m_Wk, m_Wv, m_Wo;

	public:
		Attention(Linear const& wq, Linear const& wk, Linear const& wv, Linear const& wo);
		Tensor forward(Tensor const& input) const;
	};
}