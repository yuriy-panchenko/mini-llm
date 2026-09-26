#pragma once
#include "layer.h"
#include "linear.h"

namespace llm
{
	struct AttentionOutput
	{
		Matrix out;    // attn @ V
		Matrix attn;   // softmax(scores) — cached, needed for backward
	};

	struct QKVGrads
	{
		Matrix dQ, dK, dV;
	};

	AttentionOutput scaled_dot_product_attention(Matrix const& Q, Matrix const& K, Matrix const& V, bool causal = false);

	QKVGrads scaled_dot_product_attention_backward(Matrix const& Q, Matrix const& K, Matrix const& V,
		Matrix const& attn, Matrix const& dOut, bool causal);

	Matrix sinusoidal_positional_encoding(size_t seqLen, size_t dModel);

	scalar cross_entropy(Matrix const& logits, std::vector<size_t> const& targets);

	Matrix cross_entropy_backward(Matrix const& logits, std::vector<size_t> const& targets);

	class Attention
		:public Layer
	{
		Linear m_Wq, m_Wk, m_Wv, m_Wo;
		bool m_Causal;
		Matrix m_LastQ, m_LastK, m_LastV, m_LastAttn;

	public:
		Attention(Linear const& wq, Linear const& wk, Linear const& wv, Linear const& wo, bool causal = false);
		Matrix forward(Matrix const& input);
		Matrix backward(Matrix const& dOut);

		void update(scalar lr) override;
		void zero_grad() override;
	};
}