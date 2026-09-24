#include "pch.h"
#include "multihead_attention.h"
#include "tensor.h"

namespace llm
{
	MultiHeadAttention::MultiHeadAttention(Linear const& wq, Linear const& wk, Linear const& wv, Linear const& wo, size_t numHeads)
		:m_Wq{ wq }
		, m_Wk{ wk }
		, m_Wv{ wv }
		, m_Wo{ wo }
		, m_NumHeads{ numHeads }
	{}
	
    Matrix MultiHeadAttention::forward(Matrix const& input) const
	{
        auto const Q{ m_Wq.forward(input) };
        auto const K{ m_Wk.forward(input) };
        auto const V{ m_Wv.forward(input) };

        auto const dModel{ Q.cols() };
        auto const dHead{ dModel / m_NumHeads };

        std::vector<Matrix> headOutputs;
        headOutputs.reserve(m_NumHeads);

        for (size_t h = 0; h < m_NumHeads; ++h)
        {
            auto const Qh{ Q.slice_cols(h * dHead, dHead) };
            auto const Kh{ K.slice_cols(h * dHead, dHead) };
            auto const Vh{ V.slice_cols(h * dHead, dHead) };

            auto const dK{ static_cast<scalar>(dHead) };
            auto const scores{ Qh.matmul(Kh.transpose()) / std::sqrt(dK) };
            auto const attn{ scores.softmax() };

            headOutputs.push_back(attn.matmul(Vh));
        }

        return m_Wo.forward(Matrix::concat_cols(headOutputs));
	}
}