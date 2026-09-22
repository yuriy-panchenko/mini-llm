#include "pch.h"
#include "tensor.h"

namespace llm
{
	Tensor::Tensor(std::initializer_list<size_t> const& dim)
		:m_Dim{ dim.begin(), dim.end() }
	{
		for (auto iter{ m_Dim.begin() }; iter != m_Dim.end();)
			if (*iter)
				++iter;
			else iter = m_Dim.erase(iter);

		if (!m_Dim.empty())
		{
			size_t total{ 1ull };

			for (auto u : m_Dim)
				total *= u;

			m_Data.resize(total, {});
		}
	}
	
	size_t Tensor::to_index(std::initializer_list<size_t> const& adr) const
	{
		size_t ret{};

		for (size_t i = 0; i < adr.size(); ++i)
		{
			ret *= m_Dim[i];
			ret += *(adr.begin() + i);
		}

		return ret;
	}
}
