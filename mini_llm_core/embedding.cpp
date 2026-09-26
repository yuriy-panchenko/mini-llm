// embedding.cpp
#include "pch.h"
#include "embedding.h"

namespace llm
{
	Embedding::Embedding(Matrix const& table)
		:m_Table{ table }
		, m_dTable{ table.rows(),table.cols() }
	{}

	Matrix Embedding::forward(std::vector<size_t> const& ids)
	{
		m_LastIds = ids;
		Matrix ret{ ids.size(), m_Table.cols() };

		for (size_t i = 0; i < ids.size(); ++i)
		{
			auto const src{ m_Table.row(ids[i]) };
			auto const dst{ ret.row(i) };
			std::copy(src.begin(), src.end(), dst.begin());
		}

		return ret;
	}

	void Embedding::backward(Matrix const& dOut)
	{
		// scatter-add
		for (size_t i = 0; i < m_LastIds.size(); ++i)
		{
			size_t id = m_LastIds[i];

			for (size_t c = 0; c < m_Table.cols(); ++c)
				m_dTable.at(id, c) += dOut.at(i, c);
		}
	}

	void Embedding::update(scalar lr)
	{
		for (size_t r = 0; r < m_Table.rows(); ++r)
		{
			auto dst{ m_Table.row(r) };
			std::transform(dst.begin(), dst.end(), m_dTable.row(r).begin(), dst.begin(), [lr](scalar a, scalar b)->scalar {return a - lr * b; });
		}
	}

	void Embedding::zero_grad()
	{
		m_dTable.fill({});
	}
}