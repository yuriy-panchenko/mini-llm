#pragma once
#include "transformer_block.h"

namespace llm
{
	// Chains token embedding + sinusoidal positional encoding -> a stack of
	// TransformerBlock<AttnT> -> a final Linear projection to vocab-sized logits.
	// Softmax/loss is deliberately left to the caller: forward() returns raw logits.
	template<typename AttnT>
	class Model
	{
		Embedding m_TokenEmbedding;
		std::vector<TransformerBlock<AttnT>> m_Blocks;
		Linear m_OutputHead;   // dModel -> vocabSize

	public:
		Model(Embedding const& tokenEmbedding, std::vector<TransformerBlock<AttnT>> const& blocks, Linear const& outputHead)
			:m_TokenEmbedding{ tokenEmbedding }
			, m_Blocks{ blocks }
			, m_OutputHead{ outputHead }
		{}

		Model(Embedding&& tokenEmbedding, std::vector<TransformerBlock<AttnT>>&& blocks, Linear&& outputHead)
			:m_TokenEmbedding{ std::move(tokenEmbedding) }
			, m_Blocks{ std::move(blocks) }
			, m_OutputHead{ std::move(outputHead) }
		{}

		Matrix forward(std::vector<size_t> const& ids)
		{
			auto const tokenEmb{ m_TokenEmbedding.forward(ids) };
			auto x{ tokenEmb + sinusoidal_positional_encoding(ids.size(), tokenEmb.cols()) };

			for (auto& block : m_Blocks)
				x = block.forward(x);

			return m_OutputHead.forward(x);   // seqLen x vocabSize logits
		}
	};
}