#include "pch.h"
#include "transformer_block.h"

namespace llm
{
	TransformerBlock::TransformerBlock(Attention const& attn, RMSNorm const& norm1,
		Linear const& ffn1, Linear const& ffn2, RMSNorm const& norm2)
		:m_Attn{ attn }
		, m_Norm1{ norm1 }
		, m_Ffn1{ ffn1 }
		, m_Ffn2{ ffn2 }
		, m_Norm2{ norm2 }
	{}

	Matrix TransformerBlock::forward(Matrix const& input) const
	{
#ifdef _DEBUG
		auto const normed1{ m_Norm1.forward(input + m_Attn.forward(input)) };
		auto const ffnHidden{ m_Ffn1.forward(normed1).gelu() };
		auto const ffnOut{ m_Ffn2.forward(ffnHidden) };
		return m_Norm2.forward(normed1 + ffnOut);
#else
		auto const normed1{ m_Norm1.forward(input + m_Attn.forward(input)) };
		return m_Norm2.forward(normed1 + m_Ffn2.forward(m_Ffn1.forward(normed1).gelu()));
#endif // _DEBUG
	}
}