#include "pch.h"
#include "linear.h"

namespace llm
{
	Tensor Linear::forward(const Tensor& input) const
	{
		auto ret{ matmul(input, m_Ws) };
		auto const
			rows{ m_Ws.shape()[1] },
			cols{ m_Ws.shape()[0] };
		assert(m_Bias.size() == cols);

		for (size_t iRow = 0; iRow < rows; ++iRow)
			for (size_t iCol = 0; iCol < cols; ++iCol)
				ret.at({ iRow,iCol }) += m_Bias[iCol];

		return ret;
	}
}