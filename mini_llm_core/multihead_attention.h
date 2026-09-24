#pragma once
#include "linear.h"

namespace llm
{
	class MultiHeadAttention
	{
		Linear m_Wq, m_Wk, m_Wv, m_Wo;
		size_t m_NumHeads;
		bool m_Causal;

	public:
		MultiHeadAttention(Linear const& wq, Linear const& wk, Linear const& wv, Linear const& wo, size_t numHeads, bool causal = false);
		Matrix forward(Matrix const& input) const;
	};
}