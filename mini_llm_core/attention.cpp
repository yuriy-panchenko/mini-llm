#include "pch.h"
#include "attention.h"
#include <limits>

namespace llm
{
	AttentionOutput scaled_dot_product_attention(Matrix const& Q, Matrix const& K, Matrix const& V, bool causal)
	{
		auto const dK{ static_cast<scalar>(K.cols()) };
		auto scores{ Q.matmul(K.transpose()) / std::sqrt(dK) };

		if (causal)
		{
			auto const rows{ scores.rows() }, cols{ scores.cols() };
			auto const negInf{ -std::numeric_limits<scalar>::infinity() };

			for (size_t r = 0; r < rows; ++r)
				for (size_t c = r + 1; c < cols; ++c)
					scores.at(r, c) = negInf;
		}

		auto attn{ scores.softmax() };
		auto out{ attn.matmul(V) };

		return { out, attn };
	}

	QKVGrads scaled_dot_product_attention_backward(Matrix const& Q, Matrix const& K, Matrix const& V,
		Matrix const& attn, Matrix const& dOut, bool causal)
	{
		// out = attn @ V
		auto const dAttn{ dOut.matmul(V.transpose()) };
		auto dV{ attn.transpose().matmul(dOut) };

		// attn = softmax(scores)
		auto dScores{ Matrix::d_softmax(attn, dAttn) };

		if (causal)   // masked entries were a constant in forward -- zero gradient, not "near zero"
		{
			auto const rows{ dScores.rows() }, cols{ dScores.cols() };
			for (size_t r = 0; r < rows; ++r)
				for (size_t c = r + 1; c < cols; ++c)
					dScores.at(r, c) = 0.f;
		}

		// scores = Q @ Kᵀ / sqrt(dK)
		auto const dKScale{ static_cast<scalar>(K.cols()) };
		auto const dRaw{ dScores / std::sqrt(dKScale) };

		auto dQ{ dRaw.matmul(K) };
		auto dK{ dRaw.transpose().matmul(Q) };

		return { dQ, dK, dV };
	}

	Matrix sinusoidal_positional_encoding(size_t seqLen, size_t dModel)
	{
		// unchanged
		Matrix pe{ seqLen, dModel };
		for (size_t pos = 0; pos < seqLen; ++pos)
			for (size_t i = 0; i < dModel; i += 2)
			{
				scalar const freq{ std::pow(10000.0f, -static_cast<scalar>(i) / static_cast<scalar>(dModel)) };
				pe.at(pos, i) = std::sin(static_cast<scalar>(pos) * freq);
				if (i + 1 < dModel)
					pe.at(pos, i + 1) = std::cos(static_cast<scalar>(pos) * freq);
			}
		return pe;
	}

	scalar cross_entropy(Matrix const& logits, std::vector<size_t> const& targets)
	{
		// unchanged
		assert(logits.rows() == targets.size());
		assert(logits.cols() > 0);
		scalar loss = 0.f;
		for (size_t t = 0; t < targets.size(); ++t)
		{
			auto const row{ logits.row(t) };
			const scalar max_logit{ *std::max_element(row.begin(), row.end()) };
			scalar sum_exp = 0.f;
			for (auto val : row)
				sum_exp += std::exp(val - max_logit);
			loss -= row[targets[t]] - max_logit - std::log(sum_exp);
		}
		return loss / static_cast<scalar>(targets.size());
	}

	// attention.cpp

	Matrix cross_entropy_backward(Matrix const& logits, std::vector<size_t> const& targets)
	{
		assert(logits.rows() == targets.size());
		assert(logits.cols() > 0);

		auto probs{ logits.softmax() };   // reuses the existing numerically-stable softmax
		
		for (size_t r = 0; r < targets.size(); ++r)
			probs.at(r, targets[r]) -= 1.f;

		probs /= static_cast<scalar>(targets.size());   // uses operator/=(scalar) already on Matrix

		return probs;
	}

	Attention::Attention(Linear const& wq, Linear const& wk, Linear const& wv, Linear const& wo, bool causal)
		:m_Wq{ wq }
		, m_Wk{ wk }
		, m_Wv{ wv }
		, m_Wo{ wo }
		, m_Causal{ causal }
		, m_LastQ{ 0, 0 }
		, m_LastK{ 0, 0 }
		, m_LastV{ 0, 0 }
		, m_LastAttn{ 0, 0 }
	{}

	Matrix Attention::forward(Matrix const& input)
	{
		m_LastQ = m_Wq.forward(input);
		m_LastK = m_Wk.forward(input);
		m_LastV = m_Wv.forward(input);

		auto attnOut{ scaled_dot_product_attention(m_LastQ, m_LastK, m_LastV, m_Causal) };
		m_LastAttn = attnOut.attn;

		return m_Wo.forward(attnOut.out);
	}

	Matrix Attention::backward(Matrix const& dOut)
	{
		auto dAttnOut{ m_Wo.backward(dOut) };

		auto grads{ scaled_dot_product_attention_backward(m_LastQ, m_LastK, m_LastV, m_LastAttn, dAttnOut, m_Causal) };

		auto dInput{ m_Wq.backward(grads.dQ) };
		dInput += m_Wk.backward(grads.dK);
		dInput += m_Wv.backward(grads.dV);

		return dInput;
	}
	
	void Attention::update(scalar lr)
	{
		m_Wq.update(lr);
		m_Wk.update(lr);
		m_Wv.update(lr);
		m_Wo.update(lr);
	}

	void Attention::zero_grad()
	{
		m_Wq.zero_grad();
		m_Wk.zero_grad();
		m_Wv.zero_grad();
		m_Wo.zero_grad();
	}
}