#include "pch.h"
#include "attention.h"
#include <limits>

namespace llm
{
	Matrix scaled_dot_product_attention(Matrix const& Q, Matrix const& K, Matrix const& V, bool causal)
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

		auto const attn{ scores.softmax() };

		return attn.matmul(V);
	}

	Matrix sinusoidal_positional_encoding(size_t seqLen, size_t dModel)
	{
		Matrix pe{ seqLen, dModel };

		for (size_t pos = 0; pos < seqLen; ++pos)
		{
			for (size_t i = 0; i < dModel; i += 2)
			{
				scalar const freq{ std::pow(10000.0f, -static_cast<scalar>(i) / static_cast<scalar>(dModel)) };
				pe.at(pos, i) = std::sin(static_cast<scalar>(pos) * freq);

				if (i + 1 < dModel)
					pe.at(pos, i + 1) = std::cos(static_cast<scalar>(pos) * freq);
			}
		}

		return pe;
	}

	scalar cross_entropy(Matrix const& logits, std::vector<size_t> const& targets)
	{
		assert(logits.rows() == targets.size());
		assert(logits.cols() > 0);

		scalar loss = 0.f;

		for (size_t t = 0; t < targets.size(); ++t)
		{
			// stable log-softmax of row t
			auto const row{ logits.row(t) };
			const scalar max_logit{ *std::max_element(row.begin(),row.end()) };
			scalar sum_exp = 0.f;

			for (auto val : row)
				sum_exp += std::exp(val - max_logit);

			loss -= row[targets[t]] - max_logit - std::log(sum_exp);
		}
		return loss / static_cast<scalar>(targets.size());
	}

	Attention::Attention(Linear const& wq, Linear const& wk, Linear const& wv, Linear const& wo, bool causal)
		:m_Wq{ wq }
		, m_Wk{ wk }
		, m_Wv{ wv }
		, m_Wo{ wo }
		, m_Causal{ causal }
	{}

	Matrix Attention::forward(Matrix const& input) const
	{
		auto const Q{ m_Wq.forward(input) };
		auto const K{ m_Wk.forward(input) };
		auto const V{ m_Wv.forward(input) };

		return m_Wo.forward(scaled_dot_product_attention(Q, K, V, m_Causal));
	}
}