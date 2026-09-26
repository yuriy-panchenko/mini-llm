#pragma once
#include "matrix.h"
#include "layer.h"

namespace llm
{
	class Embedding
	{
		Matrix m_Table,   // vocabSize x dModel
			m_dTable;                 // same shape
		std::vector<size_t> m_LastIds;

	public:
		explicit Embedding(Matrix const& table);
		Matrix forward(std::vector<size_t> const& ids);
		void backward(Matrix const& dOut);
	};
}