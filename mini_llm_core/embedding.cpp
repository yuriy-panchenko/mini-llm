// embedding.cpp
#include "pch.h"
#include "embedding.h"

namespace llm
{
	Embedding::Embedding(Matrix const& table)
		:m_Table{ table }
		, m_dTable{ table.rows(),table.cols() }
		, m_mTable{ table.rows(), table.cols() }
		, m_vTable{ table.rows(), table.cols() }
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
		constexpr scalar beta1{ 0.9f }, beta2{ 0.999f }, eps{ 1e-8f };
		++m_T;
		auto const bc1{ 1 - std::pow(beta1, static_cast<scalar>(m_T)) };
		auto const bc2{ 1 - std::pow(beta2, static_cast<scalar>(m_T)) };

		for (size_t r = 0; r < m_Table.rows(); ++r)
		{
			auto w{ m_Table.row(r) }, g{ m_dTable.row(r) }, m{ m_mTable.row(r) }, v{ m_vTable.row(r) };

			for (size_t c = 0; c < m_Table.cols(); ++c)
			{
				m[c] = beta1 * m[c] + (1 - beta1) * g[c];
				v[c] = beta2 * v[c] + (1 - beta2) * g[c] * g[c];

				auto const mHat{ m[c] / bc1 };
				auto const vHat{ v[c] / bc2 };

				w[c] -= lr * mHat / (std::sqrt(vHat) + eps);
			}
		}
	}

	void Embedding::zero_grad()
	{
		m_dTable.fill({});
		// m_mTable/m_vTable intentionally NOT cleared -- persist across steps
	}
}