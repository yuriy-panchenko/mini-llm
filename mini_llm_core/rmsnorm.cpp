#include "pch.h"
#include "rmsnorm.h"

namespace llm
{
	RMSNorm::RMSNorm(Tensor const& gamma, scalar eps)
		:m_Gamma{ gamma }
		, m_Eps{ eps }
	{}

	Tensor RMSNorm::forward(Tensor const& input) const
	{
		auto const rows{ input.shape()[0] }, cols{ input.shape()[1] };
		Tensor ret{ rows, cols };

		for (size_t r = 0; r < rows; ++r)
		{
			scalar sumSq{};
			for (size_t c = 0; c < cols; ++c)
			{
				auto const x{ input.at({ r, c }) };
				sumSq += x * x;
			}

			scalar const scale{ 1.f / std::sqrt(sumSq / cols + m_Eps) };

			for (size_t c = 0; c < cols; ++c)
				ret.at({ r, c }) = input.at({ r, c }) * scale * m_Gamma[c];
		}

		return ret;
	}
}