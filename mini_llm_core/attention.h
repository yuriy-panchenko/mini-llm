#pragma once
#include "linear.h"

namespace llm
{
	class Attention
	{
		Linear m_Wq, m_Wk, m_Wv, m_Wo;

	public:
		Attention(Linear const& wq, Linear const& wk, Linear const& wv, Linear const& wo);
		Matrix forward(Matrix const& input) const;
	};
}