#include "pch.h"
#include "attention.h"

namespace llm
{
	Attention::Attention(Linear const& wq, Linear const& wk, Linear const& wv, Linear const& wo)
		:m_Wq{ wq }
		, m_Wk{ wk }
		, m_Wv{ wv }
		, m_Wo{ wo }
	{
	}

	Tensor Attention::forward(Tensor const& input) const
	{
		auto const Q{ m_Wq.forward(input) };
		auto const K{ m_Wk.forward(input) };
		auto const V{ m_Wv.forward(input) };

		auto const dK{ static_cast<scalar>(K.shape()[1]) };
		auto scores{ Q.matmul(K.transpose()) / std::sqrt(dK) };

		auto const attn{ scores.softmax() };
		auto const out{ attn.matmul(V) };

		return m_Wo.forward(out);
	}
}
