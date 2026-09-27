#pragma once
#include "matrix.h"
#include "layer.h"

namespace llm
{
	class Embedding
		:public Layer
	{
		Matrix m_Table,   // vocabSize x dModel
			m_dTable;                 // same shape
		std::vector<size_t> m_LastIds;

		// Adam moment estimates -- persist across steps, untouched by zero_grad()
		Matrix m_mTable, m_vTable;
		size_t m_T{ 0 };

	public:
		explicit Embedding(Matrix const& table);
		Matrix forward(std::vector<size_t> const& ids);
		void backward(Matrix const& dOut);

		void update(scalar lr) override;
		void zero_grad() override;
	};
}