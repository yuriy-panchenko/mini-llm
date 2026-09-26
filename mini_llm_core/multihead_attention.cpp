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

		m_LastQHeads.clear();
		m_LastKHeads.clear();
		m_LastVHeads.clear();
		m_LastAttnHeads.clear();
		m_LastQHeads.reserve(m_NumHeads);
		m_LastKHeads.reserve(m_NumHeads);
		m_LastVHeads.reserve(m_NumHeads);
		m_LastAttnHeads.reserve(m_NumHeads);

		for (size_t h = 0; h < m_NumHeads; ++h)
		{
			auto Qh{ Q.slice_cols(h * dHead, dHead) };
			auto Kh{ K.slice_cols(h * dHead, dHead) };
			auto Vh{ V.slice_cols(h * dHead, dHead) };

			auto attnOut{ scaled_dot_product_attention(Qh, Kh, Vh, m_Causal) };

			headOutputs.push_back(attnOut.out);
			m_LastAttnHeads.push_back(std::move(attnOut.attn));
			m_LastQHeads.push_back(std::move(Qh));
			m_LastKHeads.push_back(std::move(Kh));
			m_LastVHeads.push_back(std::move(Vh));
		}

		return m_Wo.forward(Matrix::concat_cols(headOutputs));
	}

	Matrix MultiHeadAttention::backward(Matrix const& dOut)
	{
		auto dConcat{ m_Wo.backward(dOut) };   // seqLen x dModel, still concatenated across heads

		auto const dHead{ m_LastQHeads[0].cols() };

		std::vector<Matrix> dQHeads, dKHeads, dVHeads;
		dQHeads.reserve(m_NumHeads);
		dKHeads.reserve(m_NumHeads);
		dVHeads.reserve(m_NumHeads);

		for (size_t h = 0; h < m_NumHeads; ++h)
		{
			auto dOutH{ dConcat.slice_cols(h * dHead, dHead) };

			auto grads{ scaled_dot_product_attention_backward(
				m_LastQHeads[h], m_LastKHeads[h], m_LastVHeads[h], m_LastAttnHeads[h], dOutH, m_Causal) };

			dQHeads.push_back(std::move(grads.dQ));
			dKHeads.push_back(std::move(grads.dK));
			dVHeads.push_back(std::move(grads.dV));
		}

		auto dQ{ Matrix::concat_cols(dQHeads) };
		auto dK{ Matrix::concat_cols(dKHeads) };
		auto dV{ Matrix::concat_cols(dVHeads) };

		auto dInput{ m_Wq.backward(dQ) };
		dInput += m_Wk.backward(dK);
		dInput += m_Wv.backward(dV);

		return dInput;
	}

	void MultiHeadAttention::update(scalar lr)
	{
		m_Wq.update(lr);
		m_Wk.update(lr);
		m_Wv.update(lr);
		m_Wo.update(lr);
	}

	void MultiHeadAttention::zero_grad()
	{
		m_Wq.zero_grad();
		m_Wk.zero_grad();
		m_Wv.zero_grad();
		m_Wo.zero_grad();
	}
}