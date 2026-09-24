// transformer_block.h
#pragma once
#include "attention.h"
#include "rmsnorm.h"
#include "matrix.h"

namespace llm
{
	template<typename AttnT>
	class TransformerBlock
	{
		AttnT m_Attn;
		RMSNorm m_Norm1;
		Linear m_Ffn1, m_Ffn2;
		RMSNorm m_Norm2;

	public:
		TransformerBlock(AttnT const& attn, RMSNorm const& norm1,
			Linear const& ffn1, Linear const& ffn2, RMSNorm const& norm2)
			:m_Attn{ attn }
			, m_Norm1{ norm1 }
			, m_Ffn1{ ffn1 }
			, m_Ffn2{ ffn2 }
			, m_Norm2{ norm2 }
		{}

		Matrix forward(Matrix const& input) const
		{
			auto const resid1{ input + m_Attn.forward(m_Norm1.forward(input)) };
			auto const resid2{ resid1 + m_Ffn2.forward(m_Ffn1.forward(m_Norm2.forward(resid1)).gelu()) };
			return resid2;
		}
	};
}