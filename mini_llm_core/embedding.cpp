// embedding.cpp
#include "pch.h"
#include "embedding.h"

namespace llm
{
	Embedding::Embedding(Matrix const& table)
		:m_Table{ table }
	{}

	Matrix Embedding::forward(std::vector<size_t> const& ids) const
	{
		Matrix ret{ ids.size(), m_Table.cols() };

		for (size_t i = 0; i < ids.size(); ++i)
		{
			auto const src{ m_Table.row(ids[i]) };
			auto const dst{ ret.row(i) };
			std::copy(src.begin(), src.end(), dst.begin());
		}

		return ret;
	}
}