#pragma once
#include "matrix.h"

namespace llm
{
	class Embedding
	{
		Matrix m_Table;   // vocabSize x dModel

	public:
		explicit Embedding(Matrix const& table);
		Matrix forward(std::vector<size_t> const& ids) const;
	};
}