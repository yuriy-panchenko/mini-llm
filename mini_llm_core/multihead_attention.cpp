#include "pch.h"
#include "multihead_attention.h"
#include "attention.h"

namespace llm
{
	MultiHeadAttention::MultiHeadAttention(Linear const& wq, Linear const& wk, Linear const& wv, Linear const& wo, size_t numHeads, bool causal)
		:m_Wq{ wq }
		, m_Wk{ wk }
		, m_Wv{ wv }
		, m_Wo{ wo }
		, m_NumHeads{ numHeads }
		, m_Causal{ causal }
	{}

	Matrix MultiHeadAttention::forward(Matrix const& input)
	{
		auto const Q{ m_Wq.forward(input) };
		auto const K{ m_Wk.forward(input) };
		auto const V{ m_Wv.forward(input) };

		auto const dModel{ Q.cols() };
		assert(dModel % m_NumHeads == 0);
		auto const dHead{ dModel / m_NumHeads };

		std::vector<Matrix> headOutputs;
		headOutputs.reserve(m_NumHeads);

		for (size_t h = 0; h < m_NumHeads; ++h)
		{
			auto const Qh{ Q.slice_cols(h * dHead, dHead) };
			auto const Kh{ K.slice_cols(h * dHead, dHead) };
			auto const Vh{ V.slice_cols(h * dHead, dHead) };

			headOutputs.push_back(scaled_dot_product_attention(Qh, Kh, Vh, m_Causal));
		}

		return m_Wo.forward(Matrix::concat_cols(headOutputs));
	}
}