#include "pch.h"
#include "linear.h"

namespace llm
{
	Linear::Linear(Matrix const& w, Vector const& bias)
		:m_Ws{ w }
		, m_Bias{ bias }
	{}

	Matrix Linear::forward(const Matrix& input) const
	{
		auto ret{ input.matmul(m_Ws) };
		auto const
			rows{ ret.rows() },
			cols{ ret.cols() };
		assert(m_Bias.size() == cols);

		for (size_t iRow = 0; iRow < rows; ++iRow)
			for (size_t iCol = 0; iCol < cols; ++iCol)
				ret.at(iRow, iCol) += m_Bias[iCol];

		return ret;
	}
}