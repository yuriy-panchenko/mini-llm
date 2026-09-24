#pragma once
#include "linear.h"

namespace llm
{
	Matrix scaled_dot_product_attention(Matrix const& Q, Matrix const& K, Matrix const& V, bool causal = false);
	Matrix sinusoidal_positional_encoding(size_t seqLen, size_t dModel);

	class Attention
	{
		Linear m_Wq, m_Wk, m_Wv, m_Wo;
		bool m_Causal;

	public:
		Attention(Linear const& wq, Linear const& wk, Linear const& wv, Linear const& wo, bool causal = false);
		Matrix forward(Matrix const& input) const;
	};
}