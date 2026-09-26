#pragma once
#include "linear.h"
#include "layer.h"

namespace llm
{
	class MultiHeadAttention
		:public Layer
	{
		Linear m_Wq, m_Wk, m_Wv, m_Wo;
		size_t m_NumHeads;
		bool m_Causal;

		std::vector<Matrix> m_LastQHeads, m_LastKHeads, m_LastVHeads, m_LastAttnHeads;

	public:
		MultiHeadAttention(Linear const& wq, Linear const& wk, Linear const& wv, Linear const& wo, size_t numHeads, bool causal = false);
		Matrix forward(Matrix const& input);
		Matrix backward(Matrix const& dOut);

		// Inherited via Layer
		void update(scalar lr) override;
		void zero_grad() override;
	};
}