#include "pch.h"
#include "linear.h"

namespace llm
{
	Linear::Linear(Tensor const& w, Tensor const& bias)
		:m_Ws{ w }
		, m_Bias{ bias }
	{}

	Tensor Linear::forward(const Tensor& input) const
	{
		auto ret{ matmul(input, m_Ws) };
		auto const
			rows{ ret.shape()[0] },
			cols{ ret.shape()[1] };
		assert(m_Bias.size() == cols);

		for (size_t iRow = 0; iRow < rows; ++iRow)
			for (size_t iCol = 0; iCol < cols; ++iCol)
				ret.at({ iRow,iCol }) += m_Bias[iCol];

		return ret;
	}
}