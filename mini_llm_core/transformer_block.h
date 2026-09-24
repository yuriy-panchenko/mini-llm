#pragma once
#include "attention.h"
#include "rmsnorm.h"

namespace llm
{
	class TransformerBlock
	{
		Attention m_Attn;
		RMSNorm m_Norm1;
		Linear m_Ffn1, m_Ffn2;
		RMSNorm m_Norm2;

	public:
		TransformerBlock(Attention const& attn, RMSNorm const& norm1,
			Linear const& ffn1, Linear const& ffn2, RMSNorm const& norm2);
		Matrix forward(Matrix const& input) const;
	};
}