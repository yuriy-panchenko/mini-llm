#include "pch.h"
#include "tensor.h"

namespace llm
{
	Tensor::Tensor(std::initializer_list<size_t> const& dim)
		:m_Dim{ dim.begin(), dim.end() }
	{
		size_t total{1ull};

		for (auto u : dim)
			total *= u;

		m_Data.resize(total);
	}
}
